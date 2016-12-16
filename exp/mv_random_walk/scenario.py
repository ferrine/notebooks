from sklearn.preprocessing import StandardScaler
from pymc3.distributions.timeseries import MvGaussianRandomWalk
import theano.tensor as tt
import pandas as pd
import numpy as np
import pymc3 as pm

class Scenario(pm.Model):
    @staticmethod
    def corr(lkj):
        X = lkj[lkj.distribution.tri_index]
        X = tt.fill_diagonal(X, 1)
        return X
    
    def to_norm(self, x):
        if isinstance(x, np.ma.MaskedArray):
            data = x.filled()
            mask = x.mask
        else:
            data = x
            mask = None
        data = self.norm.transform(data) * 10
        return np.ma.MaskedArray(data, mask)
    
    def to_orig(self, x):
        if isinstance(x, np.ma.MaskedArray):
            data = x.filled()
            mask = x.mask
        else:
            data = x
            mask = None
        data = self.norm.inverse_transform(data / 10)
        return np.ma.MaskedArray(data, mask)
    
    def __init__(self, data, future=None, t=10, walk_prior=None, **kwargs):
        super(Scenario, self).__init__(name=kwargs.get('name', ''))
        if future is None:
            future = dict()
        future = {k:np.asarray(v) for k, v in future.items()}
        self.assert_all_zerodim(future)
        self.future = future
        lens = [v.size for v in future.values()]
        self.norm = StandardScaler()
        if lens and max(lens) > 1:
            self.t = max(lens)
        else:
            self.t = t
        masked, self.columns = self.get_masked(data, future, self.t, strict=kwargs.get('strict', True))
        self.norm.fit(masked[~np.isnan(masked.filled(np.nan)).any(1)])
        self.masked = self.to_norm(masked)
        if walk_prior is None:
            corr_vec = pm.LKJCorr('corr', n=kwargs.get('n', 1), p=self.p)
            sd = pm.HalfCauchy('sd', beta=kwargs.get('beta', 1), shape=(self.p,))
            cov = tt.diag(sd).dot(self.corr(corr_vec)).dot(tt.diag(sd))
            MvGaussianRandomWalk('walk', cov=cov, observed=self.masked)
        else:
            self.Var('walk', walk_prior, self.masked)
        self['walk_missing'].tag.test_value = self.missing_point
    
    @classmethod
    def get_masked(cls, data, future, t, strict=True):
        keys = set(future.keys())
        new_series = list()
        for k, series in data.to_dict('series').items():
            array = future.get(k, None)
            if array is None:
                array = future.get('d_{}'.format(k), None)
                if array is not None:
                    isdelta = True
                    keys.remove('d_{}'.format(k))
                else:
                    isdelta = None
            else:
                keys.remove(k)
                isdelta = False
            new_series.append(cls.prolonged(series, array, k, isdelta, t))
        if keys and strict:
            raise ValueError('Not all future values are used: %s' % keys)
        new_df = pd.concat(new_series, 1)
        return np.ma.MaskedArray(new_df, new_df.isnull().values), new_df.columns
        
    @property
    def p(self):
        return len(self.columns)

    @staticmethod
    def prolonged(series, array, k, isdelta, t):
        series = np.array(series)
        future = np.array([np.nan] * t)
        if array is not None:
            if array.size == 1:
                future[:] = array
            else:
                m = min(t,array.size)
                future[:m] = array[:m]
        if isdelta:
            future = future.cumsum() + series[-1]
        concated = np.concatenate([series, future])
        return pd.Series(concated, name=k)
    
    def trace_scenario(self, trace, s=1):
        mean = np.mean(trace['walk_missing'], 0)
        sd = np.std(trace['walk_missing'], 0)
        mean_df = self.masked.filled()
        upper_df = self.masked.filled()
        lower_df = self.masked.filled()
        mean_df[self.masked.mask.nonzero()] = mean
        upper_df[self.masked.mask.nonzero()] = mean + s * sd
        lower_df[self.masked.mask.nonzero()] = mean - s * sd
        return dict(
            lower=pd.DataFrame(self.to_orig(lower_df), columns=self.columns), 
            mean=pd.DataFrame(self.to_orig(mean_df), columns=self.columns),  
            upper=pd.DataFrame(self.to_orig(upper_df), columns=self.columns)
        )
    
    @staticmethod
    def filled_with_last(masked):
        masked = masked.copy()
        last = iter(masked.filled(np.nan)).__next__()
        _mask = iter(masked.mask).__next__()
        _means = masked.filled(np.nan).mean(0)
        last[_mask] = _means[_mask]
        del _mask, _means
        new_data = list()
        for row, mask in zip(masked.filled(), masked.mask):
            row[mask] = last[mask]
            new_data.append(row)
            last = row
        return np.ma.MaskedArray(new_data, masked.mask)
    
    @property
    def missing_point(self):
        fill = self.filled_with_last(self.masked).data[self.masked.mask.nonzero()]
        return fill
    
    @property
    def initial_data(self):
        return pd.DataFrame(self.to_orig(self.filled_with_last(self.masked)), columns=self.columns)
    
    @staticmethod
    def assert_all_zerodim(future):
        if not np.all([len(v.shape) <= 1 for v in future.values()]):
            raise ValueError('future values must be zero dimentional')
        else:
            pass