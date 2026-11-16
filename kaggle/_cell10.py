N_JOBS = int(globals().get('N_JOBS', 2))

from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

def _patch_numba_coverage():
    try:
        from typing import Any, Callable
        import coverage.types as ct
        if not hasattr(ct, 'Tracer'):
            ct.Tracer = ct.TracerCore
        if not hasattr(ct, 'TShouldTraceFn'):
            ct.TShouldTraceFn = Callable[..., Any]
        if not hasattr(ct, 'TShouldStartContextFn'):
            ct.TShouldStartContextFn = Callable[..., Any]
    except Exception:
        pass


_patch_numba_coverage()
try:
    import umap as umap_lib
    umap_lib.UMAP
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    umap_lib = None

NUM_RAW = [
    'Year Published', 'Min Players', 'Max Players', 'Play Time',
    'Min Age', 'Users Rated', 'Complexity Average', 'Owned Users',
]
ZERO_AS_NA = ['Year Published', 'Play Time', 'Min Age', 'Complexity Average']
NAME_RE = r'Expanded|Edition|Remastered|Legacy|Deluxe|Reprint|2nd|Second Edition'
FAMOUS_RE = (
    r'Knizia|Rosenberg|Teuber|Chv[aá]til|Chvatil|Pfister|Lacerda|'
    r'Stegmaier|Trzewiczek|Bauza|Cathala|Leacock|Garfield|Wallace|'
    r'Gerdts|Kiesling|Wehrle|Tascini|Luciani|Laukat|Childres|'
    r'Vaccarino|Konieczka|Daviau|Eklund|Kramer|Moon\b'
)
TAG_SMOOTH = 20.0
NAME_TFIDF = dict(ngram_range=(1, 2), min_df=5, max_features=3000)
NAME_SVD_COMPS = 24
TE_COLS = [
    'mean_mech_target', 'max_mech_target', 'min_mech_target', 'count_mech_target', 'var_mech_target',
    'mean_dom_target', 'max_dom_target', 'min_dom_target', 'count_dom_target', 'var_dom_target',
]


def _comma_tokens(text):
    return [t.strip() for t in text.split(',') if t.strip()]


def _norm_tags(s):
    return s.fillna('Unknown').str.replace(r'\s*,\s*', ',', regex=True)


def _tag_means(lists, y, smooth=TAG_SMOOTH):
    y = np.asarray(y, dtype=float)
    gmean = float(np.mean(y))
    sums, counts, sq = defaultdict(float), defaultdict(int), defaultdict(float)
    for tags, yi in zip(lists, y):
        for t in tags:
            sums[t] += yi
            sq[t] += yi * yi
            counts[t] += 1
    means = {t: (sums[t] + smooth * gmean) / (counts[t] + smooth) for t in counts}
    tag_var = {}
    for t, c in counts.items():
        if c > 1:
            tag_var[t] = max(sq[t] / c - (sums[t] / c) ** 2, 0.0)
        else:
            tag_var[t] = 0.0
    return means, gmean, counts, tag_var


def _tag_stats(lists, means, gmean, counts, tag_var):
    out = np.empty((len(lists), 5), dtype=float)
    for i, tags in enumerate(lists):
        known = [t for t in tags if t in means]
        if not known:
            out[i] = [gmean, gmean, gmean, 0.0, 0.0]
            continue
        vals = [means[t] for t in known]
        out[i, 0] = np.mean(vals)
        out[i, 1] = np.max(vals)
        out[i, 2] = np.min(vals)
        out[i, 3] = float(np.mean([counts.get(t, 0) for t in known]))
        out[i, 4] = float(np.mean([tag_var.get(t, 0.0) for t in known]))
    return out


class BoardGameFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        mech_max_features=150,
        dom_max_features=50,
        min_df=10,
        name_svd_components=NAME_SVD_COMPS,
        umap_components=8,
        umap_n_neighbors=15,
    ):
        self.mech_max_features = mech_max_features
        self.dom_max_features = dom_max_features
        self.min_df = min_df
        self.name_svd_components = name_svd_components
        self.umap_components = umap_components
        self.umap_n_neighbors = umap_n_neighbors

    def _as_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def _engineer_num(self, X):
        n = X[NUM_RAW].apply(pd.to_numeric, errors='coerce')
        swap = n['Min Players'] > n['Max Players']
        n.loc[swap, ['Min Players', 'Max Players']] = (
            n.loc[swap, ['Max Players', 'Min Players']].to_numpy()
        )

        year = n['Year Published']
        n['year_cent'] = (year - 2000) / 20
        n['is_old'] = ((year < 1990) & (year != 0) & year.notna()).astype(float)
        n['is_ancient'] = ((year < 1900) & (year != 0) & year.notna()).astype(float)
        n['is_modern'] = (year >= 2015).astype(float)
        n['is_year_future_or_weird'] = ((year < 0) | (year > 2026)).astype(float)
        year_missing = ((year == 0) | year.isna()).astype(float)
        complexity_missing = ((n['Complexity Average'] == 0) | n['Complexity Average'].isna()).astype(float)

        n['owned_per_rated'] = n['Owned Users'] / (n['Users Rated'] + 1)
        n['age_per_complexity'] = n['Min Age'] / (n['Complexity Average'] + 0.1)
        n['time_per_min_player'] = n['Play Time'] / (n['Min Players'].clip(lower=0) + 0.1)

        for c in ZERO_AS_NA:
            n.loc[n[c] == 0, c] = np.nan

        n['year_missing'] = year_missing
        n['complexity_missing'] = complexity_missing
        n['is_solo'] = (n['Min Players'] <= 1).astype(float)
        n['player_range'] = (n['Max Players'] - n['Min Players']).clip(lower=0)
        n['is_remastered'] = X['is_remastered'].astype(float)
        n['has_famous_designer'] = X['Name'].str.contains(FAMOUS_RE, case=False, na=False).astype(float)
        n['log_users'] = np.log1p(n['Users Rated'].clip(lower=0))
        n['log_owned'] = np.log1p(n['Owned Users'].clip(lower=0))
        n['log_time'] = np.log1p(n['Play Time'].clip(lower=0))
        n['log_users_per_owned'] = n['log_users'] - n['log_owned']
        n['complexity_sq'] = n['Complexity Average'] ** 2
        n['complexity_x_log_time'] = n['Complexity Average'] * n['log_time']
        n['playtime_per_player'] = n['log_time'] / (n['Max Players'].clip(lower=1) + 0.1)

        mech_txt = _norm_tags(X['Mechanics'])
        dom_txt = _norm_tags(X['Domains'])
        n['mech_n_tags'] = mech_txt.apply(lambda s: len(_comma_tokens(s)))
        n['dom_n_tags'] = dom_txt.apply(lambda s: len(_comma_tokens(s)))
        return n

    def _fit_mlb(self, X):
        mech_lists = _norm_tags(X['Mechanics']).map(_comma_tokens)
        dom_lists = _norm_tags(X['Domains']).map(_comma_tokens)
        mech_counts = Counter(t for tags in mech_lists for t in tags)
        dom_counts = Counter(t for tags in dom_lists for t in tags)
        self.mech_keep_ = {
            lab for lab, n in mech_counts.most_common(self.mech_max_features)
            if n >= self.min_df
        }
        self.dom_keep_ = {
            lab for lab, n in dom_counts.most_common(self.dom_max_features)
            if n >= 1
        }
        self.mlb_mech_ = MultiLabelBinarizer(classes=sorted(self.mech_keep_))
        self.mlb_dom_ = MultiLabelBinarizer(classes=sorted(self.dom_keep_))
        self.mlb_mech_.fit(mech_lists.map(lambda xs: [t for t in xs if t in self.mech_keep_]))
        self.mlb_dom_.fit(dom_lists.map(lambda xs: [t for t in xs if t in self.dom_keep_]))
        self.mech_cols_ = [f'Mech_{c}' for c in self.mlb_mech_.classes_]
        self.dom_cols_ = [f'Domain_{c}' for c in self.mlb_dom_.classes_]
        return mech_lists, dom_lists

    def _mlb_frame(self, X):
        mech_lists = _norm_tags(X['Mechanics']).map(
            lambda s: [t for t in _comma_tokens(s) if t in self.mech_keep_]
        )
        dom_lists = _norm_tags(X['Domains']).map(
            lambda s: [t for t in _comma_tokens(s) if t in self.dom_keep_]
        )
        mlb_mech = pd.DataFrame(
            self.mlb_mech_.transform(mech_lists),
            columns=self.mech_cols_,
            index=X.index,
        )
        mlb_dom = pd.DataFrame(
            self.mlb_dom_.transform(dom_lists),
            columns=self.dom_cols_,
            index=X.index,
        )
        return mlb_mech, mlb_dom

    def _num_frame(self, X):
        return pd.DataFrame(
            self.num_imputer_.transform(self._engineer_num(X)),
            columns=self.num_cols_,
            index=X.index,
        )

    def fit(self, X, y=None):
        X = self._as_df(X)
        y = np.asarray(y)
        self.feature_names_in_ = np.asarray(X.columns)

        num = self._engineer_num(X)
        self.num_cols_ = list(num.columns)
        self.num_imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        self.num_imputer_.fit(num)

        mech_lists, dom_lists = self._fit_mlb(X)
        self.mech_means_, self.mech_gmean_, self.mech_counts_, self.mech_var_ = _tag_means(mech_lists, y)
        self.dom_means_, self.dom_gmean_, self.dom_counts_, self.dom_var_ = _tag_means(dom_lists, y)

        names = X['Name'].fillna('').astype(str)
        self.tfidf_name_ = TfidfVectorizer(**NAME_TFIDF)
        tf = self.tfidf_name_.fit_transform(names)
        self.name_ridge_ = Ridge(alpha=10)
        self.name_ridge_.fit(tf, y)
        n_svd = min(self.name_svd_components, max(tf.shape[1] - 1, 1), max(tf.shape[0] - 1, 1))
        self.name_svd_ = TruncatedSVD(n_components=n_svd, random_state=42)
        name_svd = self.name_svd_.fit_transform(tf)
        self.name_svd_cols_ = [f'name_svd_{i}' for i in range(n_svd)]

        mech_te = _tag_stats(mech_lists, self.mech_means_, self.mech_gmean_, self.mech_counts_, self.mech_var_)
        dom_te = _tag_stats(dom_lists, self.dom_means_, self.dom_gmean_, self.dom_counts_, self.dom_var_)
        geom = self._geom_matrix(X, mech_te, dom_te, name_svd)
        self.umap_scaler_ = StandardScaler()
        g = self.umap_scaler_.fit_transform(geom)
        n_comp = min(self.umap_components, g.shape[1], g.shape[0] - 1)
        n_nbr = min(self.umap_n_neighbors, max(g.shape[0] - 1, 2))
        if HAS_UMAP:
            self.umap_ = umap_lib.UMAP(
                n_components=n_comp,
                n_neighbors=n_nbr,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                n_jobs=N_JOBS,
            )
        else:
            # UMAP недоступен (numba); PCA по тому же блоку числовых+TE+LSA
            self.umap_ = PCA(n_components=n_comp, random_state=42)
        self.umap_.fit(g)
        emb = getattr(self.umap_, 'embedding_', None)
        n_out = np.asarray(emb).shape[1] if emb is not None else n_comp
        self.umap_cols_ = [f'manifold_{i}' for i in range(n_out)]
        return self

    def _geom_matrix(self, X, mech_te, dom_te, name_svd):
        num = self._num_frame(X)
        te = pd.DataFrame(np.hstack([mech_te, dom_te]), columns=TE_COLS, index=X.index)
        svd = pd.DataFrame(name_svd, columns=self.name_svd_cols_, index=X.index)
        return pd.concat([num, te, svd], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)

    def _pack(self, X, mech_te, dom_te, name_pred):
        num = self._num_frame(X)
        te = pd.DataFrame(np.hstack([mech_te, dom_te]), columns=TE_COLS, index=X.index)
        mlb_mech, mlb_dom = self._mlb_frame(X)
        name_tf = self.tfidf_name_.transform(X['Name'].fillna('').astype(str))
        name_svd = self.name_svd_.transform(name_tf)
        svd_df = pd.DataFrame(name_svd, columns=self.name_svd_cols_, index=X.index)
        name_df = pd.DataFrame({'oof_name_ridge': name_pred}, index=X.index)
        geom = self._geom_matrix(X, mech_te, dom_te, name_svd)
        umap_x = self.umap_.transform(self.umap_scaler_.transform(geom.to_numpy()))
        umap_df = pd.DataFrame(umap_x, columns=self.umap_cols_, index=X.index)
        out = pd.concat([num, te, mlb_mech, mlb_dom, svd_df, name_df, umap_df], axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

    def transform(self, X):
        X = self._as_df(X)
        mech_lists = _norm_tags(X['Mechanics']).map(_comma_tokens)
        dom_lists = _norm_tags(X['Domains']).map(_comma_tokens)
        mech_te = _tag_stats(
            mech_lists, self.mech_means_, self.mech_gmean_, self.mech_counts_, self.mech_var_
        )
        dom_te = _tag_stats(
            dom_lists, self.dom_means_, self.dom_gmean_, self.dom_counts_, self.dom_var_
        )
        name_pred = self.name_ridge_.predict(
            self.tfidf_name_.transform(X['Name'].fillna('').astype(str))
        )
        return self._pack(X, mech_te, dom_te, name_pred)

    def fit_transform(self, X, y=None):
        X = self._as_df(X)
        y = np.asarray(y, dtype=float)
        self.fit(X, y)

        mech_lists = _norm_tags(X['Mechanics']).map(_comma_tokens).reset_index(drop=True)
        dom_lists = _norm_tags(X['Domains']).map(_comma_tokens).reset_index(drop=True)
        names = X['Name'].fillna('').astype(str).to_numpy()
        n = len(X)
        mech_oof = np.zeros((n, 5))
        dom_oof = np.zeros((n, 5))
        name_oof = np.zeros(n)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for tr, va in kf.split(np.arange(n)):
            m_means, m_g, m_c, m_v = _tag_means(mech_lists.iloc[tr], y[tr])
            d_means, d_g, d_c, d_v = _tag_means(dom_lists.iloc[tr], y[tr])
            mech_oof[va] = _tag_stats(mech_lists.iloc[va], m_means, m_g, m_c, m_v)
            dom_oof[va] = _tag_stats(dom_lists.iloc[va], d_means, d_g, d_c, d_v)
            vec = TfidfVectorizer(**NAME_TFIDF)
            ridge = Ridge(alpha=10)
            ridge.fit(vec.fit_transform(names[tr]), y[tr])
            name_oof[va] = ridge.predict(vec.transform(names[va]))
        return self._pack(X, mech_oof, dom_oof, name_oof)


train = load_bgg('train_data.csv')
train['is_remastered'] = train['Name'].str.contains(NAME_RE, case=False, na=False).astype(int)

features = NUM_RAW + ['Name', 'Domains', 'Mechanics', 'is_remastered']
X = train[features]
y = train['Rating Average']
sample_w = np.log1p(X['Users Rated'].clip(lower=0))

feat_kwargs = dict(
    mech_max_features=150,
    dom_max_features=50,
    min_df=10,
    name_svd_components=NAME_SVD_COMPS,
    umap_components=8,
    umap_n_neighbors=15,
)


def make_lgbm(n_estimators=2000, **kwargs):
    params = dict(
        objective='regression',
        n_estimators=n_estimators,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
        n_jobs=N_JOBS,
    )
    params.update(kwargs)
    return LGBMRegressor(**params)


def make_catboost(iterations=2000):
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        thread_count=N_JOBS,
        verbose=False,
        allow_writing_files=False,
    )


def report(name, y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'{name}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}')
    return rmse, mae, r2


cv = KFold(n_splits=5, shuffle=True, random_state=42)
print('manifold:', 'UMAP' if HAS_UMAP else 'PCA (UMAP/numba недоступен)')

print('RandomizedSearch LightGBM на фичах (LSA+manifold)...')
feat_gs = BoardGameFeatures(**feat_kwargs)
X_gs = feat_gs.fit_transform(X, y)
lgbm_search = make_lgbm(n_estimators=350)
param_dist = {
    'num_leaves': [15, 31, 47],
    'min_child_samples': [20, 30, 50],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'learning_rate': [0.015, 0.02, 0.03],
    'reg_lambda': [0.5, 1.0, 3.0],
    'subsample': [0.7, 0.85],
}
search = RandomizedSearchCV(
    lgbm_search,
    param_distributions=param_dist,
    n_iter=6,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=1,
    refit=True,
)
search.fit(X_gs, y, sample_weight=sample_w)
best_lgbm_params = dict(search.best_params_)
print('best LightGBM params:', best_lgbm_params)
print('best CV RMSE:', -search.best_score_)

lgbm_rmse, lgbm_mae, lgbm_r2, lgbm_best_iters = [], [], [], []
cb_rmse, cb_mae, cb_r2, cb_best_iters = [], [], [], []
blend_rmse, blend_mae, blend_r2 = [], [], []
oof_lgb = np.zeros(len(X))
oof_cb = np.zeros(len(X))

for fold, (tr, va) in enumerate(cv.split(X), 1):
    feat = BoardGameFeatures(**feat_kwargs)
    Xtr = feat.fit_transform(X.iloc[tr], y.iloc[tr])
    Xva = feat.transform(X.iloc[va])
    ytr, yva = y.iloc[tr], y.iloc[va]
    wtr = sample_w.iloc[tr]

    lgbm = make_lgbm(**best_lgbm_params)
    lgbm.fit(
        Xtr, ytr,
        sample_weight=wtr,
        eval_X=Xva,
        eval_y=yva,
        callbacks=[early_stopping(100, verbose=False)],
    )
    p_lgb = lgbm.predict(Xva)

    cb = make_catboost()
    cb.fit(
        Xtr, ytr,
        sample_weight=wtr,
        eval_set=(Xva, yva),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    p_cb = cb.predict(Xva)
    oof_lgb[va] = p_lgb
    oof_cb[va] = p_cb

    r = report(f'LightGBM fold {fold}', yva, p_lgb)
    lgbm_rmse.append(r[0]); lgbm_mae.append(r[1]); lgbm_r2.append(r[2])
    lgbm_best_iters.append(lgbm.best_iteration_)
    r = report(f'CatBoost fold {fold}', yva, p_cb)
    cb_rmse.append(r[0]); cb_mae.append(r[1]); cb_r2.append(r[2])
    cb_best_iters.append(cb.get_best_iteration() or cb.tree_count_)
    print(f'  iters LGB={lgbm_best_iters[-1]} CB={cb_best_iters[-1]}')


def summarize(title, rmse, mae, r2, iters=None):
    print()
    print(title)
    print('RMSE по фолдам:', np.array(rmse))
    print('Средний RMSE:  ', float(np.mean(rmse)))
    print('MAE по фолдам: ', np.array(mae))
    print('Средний MAE:   ', float(np.mean(mae)))
    print('R² по фолдам:  ', np.array(r2))
    print('Средний R²:    ', float(np.mean(r2)))
    if iters is not None:
        print('best_iteration:', iters, 'mean', float(np.mean(iters)))


blend_alpha = 0.6
blend_best = np.inf
for a in (0.4, 0.5, 0.6, 0.7, 0.8):
    pred_a = a * oof_lgb + (1.0 - a) * oof_cb
    rmse_a = mean_squared_error(y, pred_a) ** 0.5
    print(f'blend alpha={a:.1f} LGB: RMSE={rmse_a:.4f}')
    if rmse_a < blend_best:
        blend_best = rmse_a
        blend_alpha = a
print('chosen blend_alpha', blend_alpha)

p_blend_oof = blend_alpha * oof_lgb + (1.0 - blend_alpha) * oof_cb
for fold, (_, va) in enumerate(cv.split(X), 1):
    r = report(f'Blend fold {fold}', y.iloc[va], p_blend_oof[va])
    blend_rmse.append(r[0]); blend_mae.append(r[1]); blend_r2.append(r[2])

summarize('LightGBM', lgbm_rmse, lgbm_mae, lgbm_r2, lgbm_best_iters)
summarize('CatBoost', cb_rmse, cb_mae, cb_r2, cb_best_iters)
summarize(f'Blend {blend_alpha:.1f} LGB + {1-blend_alpha:.1f} CB', blend_rmse, blend_mae, blend_r2)

years = train['Year Published'].fillna(0)
time_tr = years < 2018
time_va = years >= 2018
print()
print('Time-based holdout: train year<2018, val year>=2018')
print('n_train', int(time_tr.sum()), 'n_val', int(time_va.sum()))

feat_time = BoardGameFeatures(**feat_kwargs)
Xtr_t = feat_time.fit_transform(X.loc[time_tr], y.loc[time_tr])
Xva_t = feat_time.transform(X.loc[time_va])
ytr_t, yva_t = y.loc[time_tr], y.loc[time_va]
wtr_t = sample_w.loc[time_tr]

lgbm_t = make_lgbm(**best_lgbm_params)
lgbm_t.fit(
    Xtr_t, ytr_t,
    sample_weight=wtr_t,
    eval_X=Xva_t,
    eval_y=yva_t,
    callbacks=[early_stopping(100, verbose=False)],
)
cb_t = make_catboost()
cb_t.fit(
    Xtr_t, ytr_t,
    sample_weight=wtr_t,
    eval_set=(Xva_t, yva_t),
    early_stopping_rounds=100,
    use_best_model=True,
)
p_lgb_t = lgbm_t.predict(Xva_t)
p_cb_t = cb_t.predict(Xva_t)
report('TimeCV LightGBM', yva_t, p_lgb_t)
report('TimeCV CatBoost', yva_t, p_cb_t)
report(f'TimeCV Blend {blend_alpha:.1f}/{1-blend_alpha:.1f}', yva_t, blend_alpha * p_lgb_t + (1.0 - blend_alpha) * p_cb_t)
