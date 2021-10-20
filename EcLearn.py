import os
import pandas as pd
import numpy  as np
import pybedtools as bt
#pd.set_option('display.max_rows', 2000)
#pd.set_option('display.max_columns', 100000)
#pd.set_option('display.width', 10000000)

import re
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, ShuffleSplit, 
                                     LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold)
from sklearn.feature_selection import f_regression
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             classification_report, make_scorer, balanced_accuracy_score,
                             precision_recall_curve, mean_squared_error, roc_auc_score, 
                             roc_curve, auc, r2_score, mean_absolute_error,
                             average_precision_score, explained_variance_score)

from scipy.stats import pearsonr, stats, linregress, t
from scipy.sparse import hstack, vstack
import statsmodels.api as sm

import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype']= 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['font.family'] = 'arial'
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
#from adjustText import adjust_text

import glob
import pysam
from EcBammanage import BamFilter
from concurrent import futures
from joblib import Parallel, delayed, dump, load

class STATE:
    def SMols(self, X,y):
        #statsmodels.regression.linear.OLS
        import statsmodels.api as sm
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        y_pre= est2.fittedvalues
        #print(est2.summary())
        return {'R2' : est2.rsquared,
                'R2_adj' : est2.rsquared_adj,
                'p_fv' : est2.f_pvalue,
                'intcoef' : est2.tvalues,
                'clf'  : est2,
                'p_tv' : est2.pvalues,
                'func' : "y={:.4f}x{:+.4f}".format(est2.params[X.columns[0]], est2.params['const']),
                'matrx' : pd.DataFrame( np.c_[ X, y, est2.fittedvalues], columns=['X','y', 'y_pre'])
        }

    def SMols2(self, X,y):
        #statsmodels.regression.linear.OLS
        import statsmodels.api as sm
        X1 = np.log(X+1)
        X2 = sm.add_constant(X1)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        y_pre= est2.fittedvalues
        #print(est2.summary())

        return {'R2' : est2.rsquared,
                'R2_adj' : est2.rsquared_adj,
                'p_fv' : est2.f_pvalue,
                'intcoef' : est2.tvalues,
                'p_tv' : est2.pvalues,
                'func' : "y={:.4f}ln(x+1){:+.4f}".format(est2.tvalues['ecDNAcounts'], est2.tvalues['const']),
                'matrx' : pd.DataFrame( np.c_[ X, y, y_pre], columns=['X','y', 'y_pre'])
        }

    def SCI(self, X, y):
        import scipy
        return scipy.stats.linregress(X, y)

    def F_reg_pvalue(self, y, y_pre):
        return f_regression(y.values.reshape(-1, 1), y_pre)

    def t_pvalue(self, X, y, y_pre, coef_):
        import scipy
        sse = np.sum((y_pre - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1]).astype(np.float)
        se  = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        t = coef_ / se
        p = np.squeeze(2 * (1 - scipy.stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1])))
        return [t, p]

    def T_pvalue(self, X, y, y_pre, clf):
        import scipy
        X2  = np.append(np.ones((X.shape[0],1)), X, axis=1).astype(float)
        MSE = np.sum((y-y_pre)**2)/float(X2.shape[0] -X2.shape[1])
        SE  = np.sqrt(MSE*(np.linalg.inv(np.dot(X2.T,X2)).diagonal()))
        T   = np.append(clf.intercept_, clf.coef_)/SE
        P   = np.squeeze(2*(1-scipy.stats.t.cdf(np.abs(T),(X2.shape[0] -X2.shape[1]))) )
        return [T, P]

    def SKLR(self, X, y):
        import scipy
        clf = LinearRegression()
        clf.fit(X, y)
        y_pre= clf.predict(X)
        R2 = clf.score(X, y)
        R2 = r2_score(y, y_pre)
        R2_adj = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
        intercept_ = clf.intercept_
        coef_ = clf.coef_

        p_Tv = self.T_pvalue(X, y, y_pre, clf)[1][1]
        p_fv = self.F_reg_pvalue(y, y_pre)[1][0]

        return {'R2' : R2,
                'R2_adj' : R2_adj,
                'p_fv' : p_fv,
                'intcoef' : coef_,
                'clf'  : clf,
                'p_tv' : p_Tv,
                'func' : "y={:.4f}x{:+.2f}".format(coef_[0], intercept_),
                'matrx' : pd.DataFrame( np.c_[ X, y, y_pre], columns=['X','y', 'y_pre'])
        }

    def GridS(self, M='enet'):
        from sklearn.model_selection import GridSearchCV, LeaveOneOut
        G = {'enet': {'estimator':ElasticNet(max_iter=1000, random_state=None),
                'parameters' : { 'alpha'  : [0.5,  1, 2, 5],
                                    'l1_ratio': [.01, .05, .1, .2, .3, .4, 0.5],
                                        'tol' : [1e-3, 1e-4]}},
            'Ridge' : {'estimator' : Ridge(),
                        'parameters' : {'alpha'  : [ 1, 2, 5, 7, 10, 20,30, 100],
                                        'tol' : [1e-3, 1e-4]}},
            }
        
        clf = GridSearchCV(G[M]['estimator'], G[M]['parameters'],
                        n_jobs=-2,
                        cv= ShuffleSplit(4) ,#LeaveOneOut(),
                        error_score = np.nan,
                        return_train_score=True,
                        refit = True)
        
        return clf

    def SKEnet(self, X0, y):

        import scipy
        clf = GridS()

        X = np.log( X0+1 )
        #X = X0
        clf.fit(X, y)
        y_pre= clf.predict(X)
        R2 = clf.score(X, y)
        R2 = r2_score(y, y_pre)
        R2_adj = 1 - (1-R2)*(len(y)-1)/(len(y)-X.shape[1]-1)
        intercept_ = clf.best_estimator_.intercept_
        coef_ = clf.best_estimator_.coef_
        p_Tv = T_pvalue(X, y, y_pre, clf.best_estimator_)[1][1]
        p_fv = F_reg_pvalue(y, y_pre)[1][0]

        return {'R2' : R2,
                'R2_adj' : R2_adj,
                'p_fv' : p_fv,
                'intcoef' : coef_,
                'p_tv' : p_Tv,
                'func' : "Enet: l1_ratio(%s) alpha:(%s)"%(clf.best_params_['l1_ratio'], clf.best_params_['alpha']),
                'matrx' : pd.DataFrame( np.c_[ X0, y, y_pre], columns=['X','y', 'y_pre'])
        }

    def ODtest(self, X):
        from sklearn import svm
        from sklearn.datasets import make_moons, make_blobs
        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from pyod.models.knn import KNN
        import time

        # Example settings
        n_samples = 12
        outliers_fraction = 0.1
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        # define outlier/anomaly detection methods to be compared
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
            ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=None)),
            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=8, contamination=outliers_fraction))]


        print(X)
        for name, algorithm in anomaly_algorithms:
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)
            print(name, y_pred)

class PLOT:
    def __init__(self, out):
        self.out=out
        self.color_ = [ '#00DE9F', '#FF33CC', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]

    def barP(self, K):
        K = K[(K.Cells != 'support_num')]
        K.set_index('Cells').T.plot(kind='bar', stacked=True)

    def RegPlot(self, *args, **kwargs):
        _d = kwargs.pop('data')
        Stat = SMols(_d[['ecDNAcounts']], _d['Thervalue'])
        rp = sns.regplot(x='X', y='y', data=Stat['matrx'],
                    line_kws={'label': "%s\n$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['func'], Stat['R2'], Stat['R2_adj'], Stat['p_fv'])},
                    scatter_kws={"s":4},
        )

    def PltPlot(self, *args, **kwargs):
        Model = args[0]
        _d = kwargs.pop('data')
        Stat = Model(_d[['ecDNAcounts']], _d['Thervalue'])
        
        label1 =  Stat['func']
        label2 = "$R^2$:%.4f $R^2(adj)$:%.4f p:%.4f"%(Stat['R2'], Stat['R2_adj'], Stat['p_fv'])

        plt.plot(Stat['matrx'].X, Stat['matrx'].y_pre,'ro-', label=label1)
        plt.plot(Stat['matrx'].X, Stat['matrx'].y,     'bo', label=label2)

        plt.legend(loc='upper left')

    def FGirid(self, xyData, plotM, OUT):
        pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
        g = sns.FacetGrid(xyData, 
                        row='Therical',
                        col="Cells",
                        sharey=False,
                        sharex=False,
                        palette='Set1',
                        #style='dark',
                        aspect=1.5,
                        legend_out=False,
                        #height=10,
                        #col_order=['TRA','TRB','IGH','IGL','IGK'],
            )
        if plotM == 'lr':
            Model = SMols
            g.map_dataframe(RegPlot)
        elif  plotM == 'loglr':
            Model = SMols2
            g.map_dataframe(PltPlot, Model)
        elif plotM == 'enet':
            Model = SKEnet
            g.map_dataframe(PltPlot, Model)

        for ax in g.axes.ravel():
            ax.legend(loc='upper left')

        g.savefig('%s.%s.pdf'%(OUT, plotM))
        plt.close()

        Stat = []
        for (_t,_c,_l), _g in  xyData.groupby(by=['Therical', 'Cells', 'Cellline'], sort=False):
            _S = Model(_g[['ecDNAcounts']], _g['Thervalue'])
            Stat.append( [_t,_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
        Stat = pd.DataFrame(Stat, columns=['Therical', 'Cells', 'Cellline', 'R2', 'R2_adj', 'p_fv'])
        Stat.to_csv('%s.%s.score.xls'%(OUT, plotM), sep='\t', index=False) 

        n = sns.relplot(x="Cells", y="R2", hue="Therical", style="Cellline", kind="line", palette='cool', data=Stat)
        n.set_xticklabels(rotation=270)
        n.savefig('%s.%s.score.R2.pdf'%(OUT, plotM))

    def chrec(self, C):
        plt.figure(figsize=(13,10))
        #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
        gc = sns.boxplot(x="#chrom", y="ecDNAcounts", hue="type", meanprops={'linestyle':'-.'},
                        data=C, palette="Set3",  fliersize=3, linewidth=1.5)
        plt.xticks(rotation='270')
        plt.savefig('./CellFit//AB.compare.pdf')

    def cellBox(self, xyData):
        xyR2   = xyData[['Cells', 'Rename', 'Group', 'Cellline', 'Platform', 'R2', 'R2_adj']]\
                    .drop_duplicates(keep='first').copy()
        #plt.figure(figsize=(13,10))
        #C['ecDNAcounts'] = np.log2(C['ecDNAcounts'] +1)
        xyR2.sort_values(by='Group',inplace=True)
        fig, ax = plt.subplots()
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        sns.boxplot(x="Group", y="R2",  meanprops={'linestyle':'-.'}, width=0.45,  
                        data=xyR2, palette='Set3',  fliersize=3, linewidth=0.8, ax=ax)
        sns.swarmplot(x="Group", y="R2", data=xyR2, palette=Col, linestyles='--', size=2.5, linewidth=.3, ax=ax)

        plt.xticks(rotation='270')
        plt.ylim(0, 1)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

    def cellBox1(self, xyData):
        xyR2   = xyData[['Cells', 'Rename', 'Group', 'Cellline', 'Platform', 'R2', 'R2_adj', 'p_fv', 'p_tv', 'func']]\
                    .drop_duplicates(keep='first').copy()

        xyR2.sort_values(by='Group',inplace=True)
        xyR2.to_csv(self.out +'.xls', sep='\t', index=False)
        import ptitprince as pt

        fig, ax = plt.subplots(figsize=(5,5))
        labels= sorted(xyR2.Cellline.unique())
        ort = 'v'
        pal = 'Set2'
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        ax=pt.half_violinplot( x="Cellline", y="R2", data = xyR2, palette = Col,
                                bw = .15, cut = 0.,scale = "area", width = .28, offset=0.12, 
                                linewidth = 0.8, 
                                inner = None, orient = ort)
        ax=sns.swarmplot(x="Cellline", y="R2",  data=xyR2, palette='Pastel1', 
                        linestyles='-', size=2.5, linewidth=.3)
        #ax=sns.stripplot( x="Cellline", y="R2", data = xyR2, palette = Col,
        #                    edgecolor = "white",size = 2.5, jitter = 0.04,
        #                    linestyles='--', linewidth=.3,
        #                    orient = ort)
        ax=sns.boxplot( x="Cellline", y="R2", data=xyR2, orient = ort, notch=False,
                        meanprops={'linestyle':'-.'}, width=0.2,
                        flierprops={'markersize':3, 'marker':'*', 'linestyle':'--', },
                        palette=Col, fliersize=3, linewidth=0.7)

        plt.xticks(rotation='270')
        plt.ylim(0.5, 0.75)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

    def FGPlot(self, *args, **kwargs):
        _d = kwargs.pop('data')
        X = kwargs.pop('X')
        yt= kwargs.pop('yt')
        yp= kwargs.pop('yp')

        P = _d[~(_d[yt].isna())] #plasmid
        G = _d[ (_d[yt].isna())] #gene

        label1 =  _d['func'].iloc[0]
        label2 = "$R^2$:%.4f p:%.4f"%(_d['R2'].iloc[0], _d['p_fv'].iloc[0])
        
        '''
        plt.plot( _d[X], _d[yp], linestyle='--', linewidth=2, color='fuchsia',
                    marker="*", markeredgecolor='#DA3B95', markerfacecolor='magenta',
                    markeredgewidth=0.2, markersize=8.5, label=label1)
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=35, marker='o' , edgecolors='#009BD2', color= 'cyan', label=label2)
        '''

        plt.plot( _d[X], _d[yp], linestyle='-', linewidth=2, color='black',
                    markersize=0, label=label1)
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=35, marker='o' , edgecolors='red', color= 'red', label=label2)

        if not G.empty:
            for _, _l in G.iterrows():
                plt.plot(_l[X], _l[yp],'b*', markersize=12)
                plt.text(_l[X], _l[yp], _l[0])
        plt.legend(loc='lower right')

    def FGPlotN(self, *args, **kwargs):
        _d = kwargs.pop('data')
        X = kwargs.pop('X')
        yt= kwargs.pop('yt')
        yp= kwargs.pop('yp')

        P = _d[~(_d[yt].isna())] #plasmid
        G = _d[ (_d[yt].isna())] #gene

        label1 =  _d['func'].iloc[0]
        label2 = "$R^2$:%.2f $p$:%.4f"%(_d['R2'].iloc[0], _d['p_fv'].iloc[0])
        label2 = "$R^2$:%.2f"%(_d['R2'].iloc[0])
        '''
        plt.plot( _d[X], _d[yp], linestyle='--', linewidth=2, color='fuchsia',
                    marker="*", markeredgecolor='#DA3B95', markerfacecolor='magenta',
                    markeredgewidth=0.2, markersize=8.5, label=label1)
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=35, marker='o' , edgecolors='#009BD2', color= 'cyan', label=label2)
        '''

        plt.plot( _d[X], _d[yp], linestyle='-', linewidth=2, color='black',
                    markersize=0) #label=label1
        plt.scatter( P[X],  P[yt], linewidths=0.2,
                     s=30, marker='o' , edgecolors='red', color= 'red', label=label2)

        if not G.empty:
            for _, _l in G.iterrows():
                plt.plot(_l[X], _l[yp],'b*', markersize=10, label='Mitochondria DNA' )
                #plt.text(_l[X], _l[yp], _l[0])
        plt.legend(loc='lower right')
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y',useOffset=False)
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='x',useOffset=False)

    def linepre(self, xyData,  R='Group', C='CELLs', X='ECfiltcounts', 
                yt='Thervalue', yp='Predict', Order=range(1,13), **kwargs):
        pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
        g = sns.FacetGrid(xyData, 
                        row=R,
                        col=C,
                        sharey=False,
                        sharex=False,
                        palette='Set1',
                        #style='dark',
                        aspect=1.2,
                        legend_out=False,
                        #height=10,
                        col_order=Order,
                        despine=False,
        )
        g.map_dataframe(self.FGPlotN, X=X, yt=yt, yp=yp)
        #g.set_axis_labels(C, R)
        for ax in g.axes.ravel():
            ax.legend(loc='lower right', frameon=False)
            ax.set_xlabel('Detected ecDNA counts')
            ax.set_ylabel('ecDNA copy number')
            #ax.set_yscale('log')
            #ax.set_xscale('log')
        g.tight_layout()
        g.savefig(self.out)
        plt.close()

    def linelm(self, xyData):
        g=sns.lmplot(x="ECfiltcounts", y="Thervalue", hue="Rename", data=xyData)
        g.savefig(self.out)
        plt.close()

    def linearRegP(self, xyData, R='Group', C='CELLs', X='ECfiltcounts', yt='Thervalue', yp='Predict', Mk='gene'):
        rowl = sorted(xyData[R].unique())
        coll = sorted(xyData[C].unique())

        fig, axs = plt.subplots(len(rowl), len(coll), figsize=( 60, 35)) 
        fig.set_alpha(0.0)
        #, figsize=(, 17), frameon=False ,  facecolor='w', edgecolor='k'
        for _r, _rr in enumerate(rowl):
            for _c, _cc in enumerate(coll):
                rcD = xyData[( (xyData[R]==_rr) & (xyData[C]==_cc) )]
                P = rcD[~(rcD[yt].isna())].copy() #plasmid
                G = rcD[ (rcD[yt].isna())].copy() #gene
                
                if P.empty: #same sample have no the number
                    continue
                l1 =  P['func'].iloc[0]
                l2 = "$R^2$:%.4f p:%.4f"%(P['R2'].iloc[0], P['p_fv'].iloc[0])
            
                axs[_r, _c].plot(  P[X],   P[yt], 'bo' , label=l1)
                axs[_r, _c].plot(rcD[X], rcD[yp], 'ro-', label=l2)

                axs[_r, _c].legend(loc='upper left')
                axs[_r, _c].title.set_text('y: %s | x: BC%s'%(_rr, _cc))

                if not G.empty:
                    axins = axs[_r, _c].inset_axes([0.6, 0.1, 0.38, 0.39]) #[left, bottom, width, height]
                    axins.plot(G[X], G[yp], 'c*-.')
                    for _xx, _l in G.groupby(by=X):
                        _ttxt = _l[Mk].str.cat(sep='\n')
                        axins.text(_xx, _l[yp].iloc[0], _ttxt, fontsize='x-small')
                    axs[_r, _c].indicate_inset_zoom(axins)
        fig.savefig(self.out,  bbox_inches='tight')
        plt.close()

    def Heatgene(self, xyData):
        g = sns.clustermap(xyData, row_cluster=False)
        g.savefig(self.out)
        plt.close()

    def ClustMap(self, xyData, _colm):
        figsize = (20,20)
        colm = _colm.copy()

        cor1 = colm.Platform.unique()
        cor1 = dict(zip(cor1, plt.cm.Set3(range(len(cor1)))))

        cor2 = colm.Cellline.unique()
        cor2 = dict(zip(cor2, self.color_[:len(cor2)]))

        colm.Platform = colm.Platform.map(cor1)
        colm.Cellline = colm.Cellline.map(cor2)

        hm = sns.clustermap(xyData,
                            method='complete',
                            metric='euclidean',
                            z_score=None,
                            figsize=figsize,
                            linewidths=0.001,
                            cmap="coolwarm",
                            center=0,
                            #fmt='.2f',
                            #square=True, 
                            #cbar=True,
                            yticklabels=True,
                            xticklabels=True,
                            vmin=-1.1,
                            vmax=1.1,
                            annot=False,
                            row_colors=colm,
                            )
        hm.savefig(self.out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

    def Cnvbox(self, xyData, x='Cellline',  y='CV'):
        #xyData = xyData[['Cellline', 'CV', 'Gini']].copy()
        import ptitprince as pt
        fig, ax = plt.subplots(figsize=(5,5))
        labels= sorted(xyData[x].unique())
        ort = 'v'
        pal = 'Set2'
        Col = sns.set_palette(sns.color_palette(self.color_[:8]))
        ax=pt.half_violinplot( x=x, y=y, data = xyData, palette = Col,
                                bw = .15, cut = 0.,scale = "area", width = .28, offset=0.17, 
                                linewidth = 0.8, 
                                inner = None, orient = ort)
        ax=sns.swarmplot(x=x, y=y,  data=xyData, palette='Pastel1', 
                        linestyles='-', size=2.5, linewidth=.3)
        ax=sns.boxplot( x=x, y=y, data=xyData, orient = ort, notch=False,
                        meanprops={'linestyle':'-.'}, width=0.3,
                        flierprops={'markersize':3, 'marker':'*', 'linestyle':'--', },
                        palette=Col, fliersize=3, linewidth=0.7)

        plt.xticks(rotation='270')
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

###################################first time################
class CellFit1:
    def catdf(self):
        TV='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3_Colon_theoretical_value.txt'
        TherVu =pd.read_csv(TV, sep='\t')
        Tcol =  TherVu.columns.drop('#chrom')
        TVmelt = pd.melt(TherVu, id_vars=['#chrom'], value_vars=Tcol,  var_name='Therical', value_name='Thervalue')
        TVmelt['Cellline'] = TVmelt.Therical.str.split('_').str[0]

        IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
        ACounts = []
        for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
            INput='%s/%s/EcMINIMAPont/05.CheakBP/BPState/All.plasmid.Keep.matrix'%(IN, i)
            INdata=pd.read_csv(INput, sep='\t')
            INdata.drop('Links', axis=1, inplace=True)
            Vcol = INdata.columns.drop(['#chrom'])
            INmelt = pd.melt(INdata, id_vars=['#chrom'], value_vars=Vcol,  var_name='Cells', value_name='ecDNAcounts')
            INmelt['Cellline'] = i
            ACounts.append(INmelt)
        ACounts = pd.concat(ACounts, axis=0, sort=False)
        xyData = ACounts.merge(TVmelt, on=['#chrom','Cellline'], how='outer')
        return xyData

    def CellRegres(self, xyData, _M):
        if _M == 'lr':
            Model = SMols
        elif  _M == 'loglr':
            Model = SMols2
        elif _M == 'enet':
            Model = SKEnet

        Stat = []
        for (_t,_c,_l), _g in  xyData.groupby(by=['Therical', 'Cells', 'Cellline'], sort=False):
            _S = Model(_g[['ecDNAcounts']], _g['Thervalue'])
            print(_S)
            K =(_S['matrx'].y- _S['matrx'].y_pre).abs().to_frame()
            print(K)
            ODtest(K)
            break

            #Stat.append( [_t,_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
        #Stat = pd.DataFrame(Stat, columns=['Therical', 'Cells', 'Cellline', 'R2', 'R2_adj', 'p_fv'])
        #Stat.to_csv('%s.%s.score.xls'%(OUT, plotM), sep='\t', index=False) 

    def CMD(self):
        opre='B.line'
        OUT='./CellFit/' + opre

        '''
        xyData = catdf()
        xyData.to_csv('%s.Plasmid_Col_PC3.xls'%OUT, sep='\t', index=False)
        FGirid(xyData, 'lr', OUT)
        '''

        A=pd.read_csv( './CellFit//A.line.Plasmid_Col_PC3.xls', sep='\t')
        B=pd.read_csv( './CellFit//B.line.Plasmid_Col_PC3.xls', sep='\t')
        '''
        R =  pd.read_csv( './CellFit//B.line.lr.score.xls', sep='\t')
        Rt = R.pivot(index='Therical', columns='Cells', values='R2')
        Rt.to_csv('./CellFit//B.line.lr.score.R2.t.xls', sep='\t')

        CellRegres(B, 'lr')
        '''

        A['type'] = 'A'
        B['type'] = 'B'
        C=pd.concat((A,B), axis=0)
        C = C[(C.Cells != 'support_num')]
        chrec(C)
        cellec(C)

###################################second time################
class CellFit2:
    CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
    PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
            '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
            'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
            'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
            'SunTag-CRISPRi', 'V7-MC-HG-FA']
    CELLS = [ 'BC%s'%i for i in range(1,13) ]
    MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
                'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
                'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
                'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']
    def getcounts(self, dfmrx):
        dfmrx.loc[(dfmrx['#chrom'].isin(PLMD)), 'gene_name'] = dfmrx.loc[(dfmrx['#chrom'].isin(PLMD)), '#chrom']

        countdict={}
        for _, _l in dfmrx.iterrows():
            if (not _l.gene_name) or (_l.gene_name=='.'):
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            for _i in list(zip(_G, _B)):
                if _i[0] !='.':
                    countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'#chrom': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        countlist = countlist[(countlist.gene_biotype.isin(['protein_coding', '.']))]
        CCol = countlist.columns.drop(['#chrom', 'gene_biotype'])
        countlist = pd.melt(countlist, id_vars=['#chrom'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts' )
        return countlist

    def getdf(self):
        TV='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3_Colon_theoretical_value.txt'
        TherVu =pd.read_csv(TV, sep='\t')
        Tcol =  TherVu.columns.drop('#chrom')
        TVmelt = pd.melt(TherVu, id_vars=['#chrom'], value_vars=Tcol,  var_name='Therical', value_name='Thervalue')
        TVmelt['Cellline'] = TVmelt.Therical.str.split('_').str[0]
        print(TVmelt)

        IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
        OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit2/'
        ACounts = []
        for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
            INput='%s/%s/EcMINIMAPont/05.CheakBP/BPState/All.plasmid.Keep.matrix'%(IN, i)
            INdata=pd.read_csv(INput, sep='\t')
            INdata.drop('Links', axis=1, inplace=True)
            Vcol = INdata.columns.drop(['#chrom'])
            INmelt = pd.melt(INdata, id_vars=['#chrom'], value_vars=Vcol,  var_name='Cells', value_name='BPcounts')
            INmelt['Cellline'] = i
            #INmelt['Datatype'] = 'BPcount'
            ACounts.append(INmelt)
        ACounts = pd.concat(ACounts, axis=0, sort=False)
        #print(ACounts)

        BCounts = []
        for i in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2']:
            UpFilter=t='%s/%s/EcMINIMAPont/04.EcRegion/All.circle.region.UpFilter'%(IN, i)
            UpFilter=pd.read_csv(UpFilter, sep='\t')
            UpFilter=UpFilter.loc[ (UpFilter.groupby(by='LINKS')['length'].idxmax()) ] # Tpye='maxlen'
            #UpFilter=UpFilter.loc[ (UpFilter.Type==1) ] # Tpye='type1'
            UpFilter=getcounts(UpFilter)
            UpFilter['Cellline'] = i
            #UpFilter['Datatype'] = 'ECfilt'
            BCounts.append(UpFilter)
        BCounts = pd.concat(BCounts, axis=0, sort=False)

        #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)
        xyData = BCounts\
                    .merge(ACounts, on=['#chrom','Cellline', 'Cells'], how='outer')\
                    .merge(TVmelt,  on=['#chrom','Cellline'], how='outer')

        xyData.to_csv('%s/EcDNA_Plasmid_Col_PC_maxlen.xls'%OU, sep='\t', index=False)
        print(xyData)

    def FGPlot(self, *args, **kwargs):
        _d = kwargs.pop('data')
        M = kwargs.pop('M')
        P = kwargs.pop('P')
        R = kwargs.pop('R')
        C = kwargs.pop('C')

        _d.columns = _d.columns.tolist()[:-2] + ['X', 'y']
        Stat = M(_d[['X']], _d['y'])

        label1 =  Stat['func']
        label2 = "$R^2$:%.4f p:%.4f"%(Stat['R2'], Stat['p_fv'])
        
        plt.plot(Stat['matrx'].X, Stat['matrx'].y_pre,'ro-', label=label1)
        plt.plot(Stat['matrx'].X, Stat['matrx'].y,    'bo', label=label2)
        if not P.empty:
            P = P[( (P[R].isin(_d[R])) & (P[C].isin(_d[C])) )]
            for _, _l in P.iterrows():
                plt.plot(_l[-2], _l[-1],'c*')
                plt.text(_l[-2], _l[-1], _l[0])
        plt.legend(loc='upper left')

    def linearReg(self, xyData, OUT, PD = pd.DataFrame(), _M='sklr', R='Therical', C='Cells', R2=False):
        if _M == 'lr':
            Model = SMols
        elif  _M == 'loglr':
            Model = SMols2
        elif _M == 'enet':
            Model = SKEnet
        elif _M == 'sklr':
            Model = SKLR
        #print(locals())
        pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
        g = sns.FacetGrid(xyData, 
                        row=R,
                        col=C,
                        sharey=False,
                        sharex=False,
                        palette='Set1',
                        #style='dark',
                        aspect=1.5,
                        legend_out=False,
                        #height=10,
                        col_order=CELLS,
        )

        g.map_dataframe(FGPlot, M=Model, P=PD, R=R, C=C)
        g.set_axis_labels(C, R)
        for ax in g.axes.ravel():
            ax.legend(loc='upper left')
            #ax.set_yscale('log')
            #ax.set_xscale('log')
        g.tight_layout()
        g.savefig(OUT + '.pdf')
        plt.close()

        Stat = []
        for (_c,_l), _g in  xyData.groupby(by=[R, C], sort=False):
            _S = Model(_g.iloc[:, [-2]], _g.iloc[:, -1])
            Stat.append( [_c,_l, _S['R2'], _S['R2_adj'], _S['p_fv']] )
        Stat = pd.DataFrame(Stat, columns=[R, C, 'R2', 'R2_adj', 'p_fv'])
        Stat['Cells'] = pd.Categorical(Stat['Cells'], CELLS)
        Stat.sort_values(by=['Cells', R], inplace=True)
        Stat.to_csv(OUT + '.xls', sep='\t', index=False) 
        
        if R2:
            n = sns.relplot(x=C, y="R2", hue=R, style=R, kind="line", palette='tab10', data=Stat)
            n.set_xticklabels(rotation=270)
            n.set(ylim=(0, 1))
            n.savefig(OUT + '.R2.pdf')

    def linearRegP(self, T, P, OUT, R='Therical', C='Cells', Xc=['ECfiltcounts'], yc='Thervalue'):
        def mstat(X, y, Xpre, _M='sklr'):
            if _M == 'lr':
                M = SMols
            elif  _M == 'loglr':
                M = SMols2
            elif _M == 'enet':
                M = SKEnet
            elif _M == 'sklr':
                M = SKLR
            S = M(X, y)
            l1 =  S['func']
            l2 = "$R^2$:%.4f p:%.4f"%(S['R2'], S['p_fv'])
            ypre = S['clf'].predict(Xpre) if len(Xpre)>0 else []
            return (l1, l2, S, ypre)

        rowl = T[R].unique()
        coll = T[C].unique()
        P = P.copy()
        P[Xc] = P[Xc].astype(int)

        fig, axs = plt.subplots(len(rowl), len(coll), figsize=( 60, 18)) 
        fig.set_alpha(0.0)
        #, figsize=(, 17), frameon=False ,  facecolor='w', edgecolor='k'
        for _r, _rr in enumerate(rowl):
            for _c, _cc in enumerate(coll):
                _tt = T[( (T[R]==_rr) & (T[C]==_cc) )]
                _bb = ((P[R]==_rr) & (P[C]==_cc))
                _pp = P[_bb]
                l1, l2, S, ypre = mstat(_tt[Xc], _tt[yc], _pp[Xc])
                P.loc[_bb, yc] = ypre

                axs[_r, _c].plot(S['matrx'].X, S['matrx'].y_pre, 'ro-', label=l1)
                axs[_r, _c].plot(S['matrx'].X, S['matrx'].y,    'bo'  , label=l2)
                axs[_r, _c].legend(loc='upper left')
                axs[_r, _c].title.set_text('y: %s | x: %s'%(_rr, _cc))

                if _bb.any():
                    axins = axs[_r, _c].inset_axes([0.57, 0.1, 0.4, 0.4]) #[left, bottom, width, height]
                    axins.plot(_pp[Xc], ypre, 'r*-.')
                    for _xx, _l in _pp.groupby(by=Xc):
                        _ttxt = _l['#chrom'].str.cat(sep='\n')
                        axins.text(_xx, _l[yc].iloc[0], _ttxt, fontsize='x-small')
                    axs[_r, _c].indicate_inset_zoom(axins)
                    '''
                    #axins.set_xlim(x1, x2)
                    #axins.set_ylim(y1, y2)
                    texts = []
                    for _xx, _l in _pp.groupby(by=Xc):
                        _ttxt = _l['#chrom'].str.cat(sep='\n')
                        texts.append( axins.text(_xx, _l[yc].iloc[0], _ttxt, fontsize='x-small') )
                    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='c', lw=0.5))
                    axs[_r, _c].indicate_inset_zoom(axins)
                    '''
        fig.savefig(OUT+'.pdf',  bbox_inches='tight')

    def linearRegC(self, xyData, OUT, _M='sklr', R='Therical', xl = 'BPcounts', yl='ECfiltcounts', R2=False):
        if _M == 'lr':
            Model = SMols
        elif  _M == 'loglr':
            Model = SMols2
        elif _M == 'enet':
            Model = SKEnet
        elif _M == 'sklr':
            Model = SKLR

        pal = dict(TRA='Set1', TRB='Set2', IGH='Set3', IGL='cool', IGK='hot' )
        g = sns.FacetGrid(xyData, 
                        col=R,
                        sharey=False,
                        sharex=False,
                        palette='Set1',
                        #style='dark',
                        aspect=1.5,
                        legend_out=False,
                        #height=10,
                        #col_order=CELLS,
        )
        g.map_dataframe(FGPlot, M=Model)
        g.set_axis_labels(xl, yl)
        for ax in g.axes.ravel():
            ax.legend(loc='upper left')
        g.tight_layout()
        g.savefig(OUT + '.pdf')
        plt.close()

    def predictCN(self, T, P, _M ='sklr',
                plasther={'Colon-P1':'Colon-P1_pikein-100', 
                            'Colon-P2':'Colon-P2_100', 
                            'PC3-P1'  :'PC3-P1_spikein-100',
                            'PC3-P2'  :'PC3-P2_P2-100'}):
        MODLES = {}
        if _M == 'lr':
            Model = SMols
        elif  _M == 'loglr':
            Model = SMols2
        elif _M == 'enet':
            Model = SKEnet
        elif _M == 'sklr':
            Model = SKLR

        T = T[( (T.Therical.isin(plasther.values())) & (T.Cellline.isin(plasther.keys())) )].copy()
        #T[['ECfiltcounts', 'Thervalue']] = T[['ECfiltcounts', 'Thervalue']].fillna(0)

        P = P.copy()
        P['ECfiltcounts'] = P['ECfiltcounts'].fillna(0)
        P['Therical'] = P.Cellline.map(plasther)

        for (_c, _l, _t), _g in  T.groupby(by=['Cells', 'Cellline', 'Therical']):
            _B = ((P.Cells==_c) & (P.Cellline==_l) & (P.Therical==_t))
            if _B.any():
                Stat = Model(_g[['ECfiltcounts']], _g['Thervalue'])
                P.loc[_B, 'Thervalue'] = Stat['clf'].predict(P.loc[_B, ['ECfiltcounts']] )
        return T, P

    def CMD2(self, M = 'sklr', Type='maxlen'):
        IN='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore'
        OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit2'
        #getdf()

        if Type == 'maxlen':
            A=pd.read_csv( '%s/EcDNA_Plasmid_Col_PC_maxlen.xls'%OU, sep='\t')
        elif Type == 'type1':
            A=pd.read_csv( '%s/EcDNA_Plasmid_Col_PC_type1.xls'%OU, sep='\t')
        
        A = A[(~A.ECfiltcounts.isna())]
        A['Cells'] = pd.Categorical(A['Cells'], CELLS+['support_num'])
        A.sort_values(by=['Cells', '#chrom'], inplace=True)


        T  = A[( A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.Thervalue.isna()) )]
        P  = A[( A['#chrom'].isin(MARKGENE) & (A.Cells.isin(CELLS)) & (A.ECfiltcounts>0) )]
        T, P = predictCN(T, P, _M=M)
        T = T[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']]
        P = P[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']]
        H = '%s/EcDNA_ECfiltvsThervalue_predict_%s_%s'%(OU, M, Type)
        pd.concat([T, P], axis=0).to_csv( H + '.xls', sep='\t', index=False)
        linearRegP(T, P, H)
        #linearReg(T, '%s/EcDNA_ECfiltvsThervalue_predict_%s_%s'%(OU, M, Type), PD=P, R='Therical', C='Cells', R2=True)

        '''
        ####BPcount vs ECfilt
        B = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~A.BPcounts.isna()) )].copy()
        B = B[['#chrom', 'Cells', 'Cellline', 'ECfiltcounts', 'BPcounts']].drop_duplicates(keep='first')
        linearReg(B, '%s/EcDNA_BPvsEcFilter_%s_%s'%(OU, M, Type), _M =M, R='Cellline', C='Cells')


        ####BPcount vs Thervalue
        C = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.BPcounts.isna()) )]
        C = C[['#chrom', 'Cells', 'Cellline', 'Therical', 'BPcounts', 'Thervalue']].drop_duplicates(keep='first')
        linearReg(C, '%s/EcDNA_BPvsThervalue_%s_%s'%(OU, M, Type), R='Therical', C='Cells', R2=True)

        ####ECfilt vs Thervalue
        D = A[(A['#chrom'].isin(PLMD) & (A.Cells.isin(CELLS)) & (~ A.Thervalue.isna()) )]
        D = D[['#chrom', 'Cells', 'Cellline', 'Therical', 'ECfiltcounts', 'Thervalue']].drop_duplicates(keep='first')
        linearReg(D, '%s/EcDNA_ECfiltvsThervalue_%s_%s'%(OU, M, Type), R='Therical', C='Cells', R2=True)
        linearRegC(D, '%s/EcDNA_ECfiltvsThervalueC_%s_%s'%(OU, M, Type), R='Therical', R2=True)
        '''

###################################third time################
class CellFit3:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
                        'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
                        'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
                        'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']

    def _getinfo(self, INF, TherVu, OU):
        INdf   = pd.read_csv(INF, sep='\t')
        self.INdf   = INdf[(INdf.Filter == 'Keep')].copy()
        self.INdf.rename(columns={"DNA": "Cells"}, inplace=True)
        self.TVmelt = TherVu
        self.outdir = OU
        self.outhead= 'allsamples.TRF.'
        self.outpre = self.outdir + '/' + self.outhead
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if (not _l.gene_name) or (_l.gene_name=='.'):
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for IN in INs:
            UpFilter=pd.read_csv(IN, sep='\t')

            PLLinks = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)

            UpFilter= self.getcounts(UpFilter)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        return BCounts

    def getdf(self, INs):
        #ACounts = self.CheakBPMT()
        BCounts = self.ecRegion(INs)
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)

        BCounts= pd.read_csv( self.outpre + 'gene.counts.txt', sep='\t' )
        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.TVmelt, on='gene', how='outer')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyData

    def getstat(self, xyData, GRP='Cells'):
        xyStat = []
        for _c, _g in xyData.groupby(GRP, sort=False):
            Train = _g[ (_g.gene.isin(self.PLMD) & (~ _g.Thervalue.isna()) & (~ _g.ECfiltcounts.isna()) )].copy()
            Pred  = _g[ (_g.gene.isin(self.MARKGENE) & (~ _g.ECfiltcounts.isna()) )].copy()
            if Train.shape[0] <3:
                continue
            State = self.TrainPre( Train[['ECfiltcounts']], Train['Thervalue'], Pred[['ECfiltcounts']])
            xyTP  = pd.concat( [Train, Pred], axis=0 )
            xyTP['Predict'] = np.r_[State['matrx']['y_pre'].values, State['predy']]
            xyTP['R2']      = State['R2']
            xyTP['R2_adj']  = State['R2_adj']
            xyTP['p_fv']    = State['p_fv']
            xyTP['p_tv']    = State['p_tv']
            xyTP['func']    = State['func']
            xyStat.append(xyTP)
        xyStat = pd.concat(xyStat,axis=0)
        xyStat.to_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyStat

    def TrainPre(self, Xtr, ytr, Xpr, _M='sklr'):
        if _M == 'lr':
            Model = STATE().SMols
        elif  _M == 'loglr':
            Model = STATE().SMols2
        elif _M == 'enet':
            Model = STATE().SKEnet
        elif _M == 'sklr':
            Model = STATE().SKLR
        Stat = Model(Xtr, ytr)
        
        if Xpr.shape[0] >0:
            Stat['predy'] = Stat['clf'].predict(Xpr)
        else:
            Stat['predy'] = np.array([])
        return Stat

    def CMD3(self, INs, INF, TherVu, OU):
        self._getinfo(INF, TherVu, OU)
        '''
        xyData = self.getdf(INs)
        xyStat = self.getstat(xyData)
        #xyData = pd.read_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', low_memory=False)
        #xyStat = pd.read_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t')
        xyStat['CELLs'] = xyStat.Cells.str.split('_BC').str[1].astype(int)
        xyStat.sort_values(by=['Group','CELLs'], inplace=True)
        PLOT(self.outpre + 'linear.gene.plasmid.counts.pdf').linepre(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.counts2.pdf').linearRegP(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.R2.pdf').cellBox(xyStat)
        '''
        geneMT = pd.read_csv(self.outpre + 'gene.counts.txt', sep='\t', low_memory=False)
        geneMT = geneMT[ (geneMT.gene_biotype.isin(['protein_coding', '.']))].fillna(0)
        geneMT.drop('gene_biotype', axis=1, inplace=True)
        geneMT.set_index('gene', inplace=True)
        #geneMT = geneMT/geneMT.sum(0)
        geneMT = geneMT[~(geneMT.index.isin(self.PLMD))]
        geneMT = geneMT[~(geneMT.index.str.contains('MT'))]
        print(geneMT.max(1).sort_values())
        PLOT(self.outpre + 'gene.counts.pdf').Heatgene(geneMT)

def Start3():
    THEORETV = [['2x35S-eYGFPuv-T878-p73', 10000],
                ['2x35S-LbCpf1-pQD', 8000],
                ['380B-eYGFPuv-d11-d15', 2000],
                ['5P2T-pKGW7', 1200],
                ['A10-pg-p221', 9000],
                ['Cas9-U6-sgRNA-pQD', 1500],
                ['HD-T878-UBQ10', 100],
                ['Lat52-grim-TE-MC9-prk6-pKGW7', 3000],
                ['Lat52-RG-HTR10-1-GFP-pBGW7', 5500],
                ['myb98-genomic-nsc-TOPO', 4000],
                ['pB2CGW', 7500],
                ['pHDzCGW', 800],
                ['pQD-in', 400],
                ['pro18-Mal480-d1S-E9t', 500],
                ['SunTag-CRISPRi', 200],
                ['V7-MC-HG-FA', 4000]]
    THEORETV = pd.DataFrame(THEORETV, columns=['gene', 'Thervalue'])
    #CELLS = [ 'BC%s'%i for i in range(1,13) ]

    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit/20210128'
    CellFit3().CMD3(IN, INF, THEORETV, OU)

###################################fourth time################
class CellFit4:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.MARKGENE = ['EGFR', 'CDK6', 'SEPTIN14', 'MYC', 'DENND3', 
                        'PCAT1', 'BAP1', 'SOX2', 'MUC4', 'MECOM', 'PIK3CA', 
                        'CCND1', 'MYCN', 'TERT', 'RPS6', 'SMARCA4', 'WDR60', 
                        'AC019257.8', 'DLG1', 'WNK1', 'MUC2', 'AHRR']
        self.MARKGENE1= ['MT-ND1']

    def _getinfo(self, INF, TherVu, OU):
        INdf   = pd.read_csv(INF, sep='\t')
        self.INdf   = INdf[(INdf.Filter == 'Keep')].copy()
        self.INdf.rename(columns={"DNA": "Cells"}, inplace=True)
        self.TVmelt = TherVu
        self.outdir = OU
        self.outhead= 'allsamples.TRF.'
        self.outpre = self.outdir + '/' + self.outhead
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if (not _l.gene_name) or (_l.gene_name=='.'):
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            #_S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            _S = dict(zip( _l.SIDs.split(';'), map( int,_l.Supports.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for IN in INs:
            UpFilter=pd.read_csv(IN, sep='\t')

            PLLinks = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLLinks))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)
            UpFilter[['gene_name', 'gene_biotype']] = UpFilter[['gene_name', 'gene_biotype']].astype(str).fillna('.')

            UpFilter= self.getcounts(UpFilter)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        return BCounts

    def getdf(self, INs):
        #ACounts = self.CheakBPMT()
        BCounts = self.ecRegion(INs)
        #CCounts = pd.concat(ACounts + BCounts, axis=0, sort=False)
        BCounts= pd.read_csv( self.outpre + 'gene.counts.txt', sep='\t' )
        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.TVmelt, on='gene', how='outer')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyData

    def TrainPre(self, Xtr, ytr, Xpr, _M='sklr'):
        if _M == 'lr':
            Model = STATE().SMols
        elif  _M == 'loglr':
            Model = STATE().SMols2
        elif _M == 'enet':
            Model = STATE().SKEnet
        elif _M == 'sklr':
            Model = STATE().SKLR

        Stat = Model(Xtr, ytr)
        
        if Xpr.shape[0] >0:
            Stat['predy'] = Stat['clf'].predict(Xpr)
        else:
            Stat['predy'] = np.array([])
        return Stat

    def getstat(self, xyData, GRP='Cells'):
        xyStat = []
        for _c, _g in xyData.groupby(GRP, sort=False):
            Train = _g[ (_g.gene.isin(self.PLMD) & (~ _g.Thervalue.isna()) & (~ _g.ECfiltcounts.isna()) )].copy()
            Pred  = _g[ (_g.gene.isin(self.MARKGENE) & (~ _g.ECfiltcounts.isna()) )].copy()
            if Train.shape[0] <3:
                continue
            State = self.TrainPre( Train[['ECfiltcounts']], Train['Thervalue'], Pred[['ECfiltcounts']])
            xyTP  = pd.concat( [Train, Pred], axis=0 )
            xyTP['Predict'] = np.r_[State['matrx']['y_pre'].values, State['predy']]
            xyTP['R2']      = State['R2']
            xyTP['R2_adj']  = State['R2_adj']
            xyTP['p_fv']    = State['p_fv']
            xyTP['p_tv']    = State['p_tv']
            xyTP['func']    = State['func']
            xyStat.append(xyTP)
        xyStat = pd.concat(xyStat,axis=0)
        xyStat.to_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyStat

    def getdf1(self, INs):
        BCounts = self.ecRegion(INs)

        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData = xyData.merge(self.TVmelt, on=['gene','Group'], how='right')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)

        return xyData

    def CMD4(self, INs, INF, TherVu, OU):
        TherVu = pd.read_csv(TherVu, sep='\t')
        self._getinfo(INF, TherVu, OU)
        self.Tcol   = TherVu.columns.drop('gene')
        self.TVmelt = pd.melt(TherVu, id_vars=['gene'], value_vars=self.Tcol, var_name='Group', value_name='Thervalue')
        xyData = self.getdf1(INs)
        xyStat = self.getstat(xyData)
        #xyData = pd.read_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', low_memory=False)

        xyStat = pd.read_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t')
        xyStat['CELLs'] = xyStat.Cells.str.split('_BC').str[1].astype(int)
        xyStat.sort_values(by=['Group','CELLs'], inplace=True)
        PLOT(self.outpre + 'linear.gene.plasmid.R2.pdf').cellBox1(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.counts.pdf').linepre(xyStat, R='Group',Order=None )

        SelectC = xyStat[['Cells', 'Cellline', 'R2']]\
                    .drop_duplicates(keep='first')\
                    .sort_values(by='R2', ascending=False)\
                    .groupby('Cellline', sort=False)\
                    .head(6)
        SelectC['neworder']=1
        SelectC['neworder'] = SelectC.groupby(by='Cellline', sort=False)['neworder'].apply(np.cumsum)
        xyStat= xyStat.merge(SelectC, on=['Cells', 'Cellline','R2'], how='right')
        xyStat.to_csv(self.outpre + 'linear.3cellline.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cellline.plasmid.counts.pdf').linepre(xyStat, R='Cellline', C='neworder', Order=None )

        xyStat = xyStat[(xyStat.Cells.isin(['Colon-P2_BC5', 'PC3-P2_BC5', 'PDACDNA2_BC5']))]
        xyStat.to_csv(self.outpre + 'linear.3cells.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cells.plasmid.counts.pdf').linepre(xyStat, R=None, C='Cells', Order=None )

    def getdf2(self, INs):
        BCounts = self.ecRegion(INs)

        xyData = BCounts[(BCounts.gene_biotype.isin(['protein_coding', '.']))]
        CCol   = xyData.columns.drop(['gene', 'gene_biotype'])
        xyData = pd.melt(xyData, id_vars=['gene'], value_vars=CCol, var_name='Cells', value_name='ECfiltcounts')
        xyData = xyData.merge(self.INdf[['Cells', 'Rename', 'Group', 'Cellline', 'Platform']], on='Cells', how='outer')
        xyData = xyData.merge(self.TVmelt, on=['gene','Group'], how='left')
        xyData.to_csv(self.outpre + 'trans.gene.plasmid.counts.txt', sep='\t', index=False)

        return xyData

    def getstat1(self, xyData, GRP='Cells'):
        xyStat = []
        for _c, _g in xyData.groupby(GRP, sort=False):
            Train = _g[ (_g.gene.isin(self.PLMD) & (~ _g.Thervalue.isna()) & (~ _g.ECfiltcounts.isna()) )].copy()
            Pred  = _g[ (_g.gene.isin(self.MARKGENE1) & (~ _g.ECfiltcounts.isna()) )].copy()
            if Train.shape[0] <3:
                continue
            State = self.TrainPre( Train[['ECfiltcounts']], Train['Thervalue'], Pred[['ECfiltcounts']])
            xyTP  = pd.concat( [Train, Pred], axis=0 )
            xyTP['Predict'] = np.r_[State['matrx']['y_pre'].values, State['predy']]
            xyTP['R2']      = State['R2']
            xyTP['R2_adj']  = State['R2_adj']
            xyTP['p_fv']    = State['p_fv']
            xyTP['p_tv']    = State['p_tv']
            xyTP['func']    = State['func']
            xyStat.append(xyTP)
        xyStat = pd.concat(xyStat,axis=0)
        xyStat.to_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t', index=False)
        return xyStat

    def CMD41(self, INs, INF, TherVu, OU):
        TherVu = pd.read_csv(TherVu, sep='\t')
        self._getinfo(INF, TherVu, OU)
        self.Tcol   = TherVu.columns.drop('gene')
        self.TVmelt = pd.melt(TherVu, id_vars=['gene'], value_vars=self.Tcol, var_name='Group', value_name='Thervalue')
        xyData = self.getdf2(INs)
        xyStat = self.getstat1(xyData)

        xyStat = pd.read_csv(self.outpre + 'linear.gene.plasmid.counts.txt', sep='\t')
        xyStat['CELLs'] = xyStat.Cells.str.split('_BC').str[1].astype(int)
        xyStat.sort_values(by=['Group','CELLs'], inplace=True)
        PLOT(self.outpre + 'linear.gene.plasmid.R2.pdf').cellBox1(xyStat)
        PLOT(self.outpre + 'linear.gene.plasmid.counts.pdf').linepre(xyStat, R='Group',Order=None )

        SelectC = xyStat[['Cells', 'Cellline', 'R2']]\
                    .drop_duplicates(keep='first')\
                    .sort_values(by='R2', ascending=False)\
                    .groupby('Cellline', sort=False)\
                    .head(6)
        SelectC['neworder']=1
        SelectC['neworder'] = SelectC.groupby(by='Cellline', sort=False)['neworder'].apply(np.cumsum)
        xyStat= xyStat.merge(SelectC, on=['Cells', 'Cellline','R2'], how='right')
        xyStat.to_csv(self.outpre + 'linear.3cellline.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cellline.plasmid.counts.pdf').linepre(xyStat, R='Cellline', C='neworder', Order=None )

        xyStat = xyStat[(xyStat.Cells.isin(['Colon-P2_BC5', 'PC3-P2_BC5', 'PDACDNA2_BC5']))]
        xyStat.to_csv(self.outpre + 'linear.3cells.plasmid.counts.xls', sep='\t', index=False)
        PLOT(self.outpre + 'linear.3cells.plasmid.counts.pdf').linepre(xyStat, R=None, C='Cells', Order=None )

def Start4():
    THEORETV = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/Colon-P2_PC3-P2_PDAC_theoretical_value.txt'
    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210121/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/CellFit/20210131'
    CellFit4().CMD4(IN, INF, THEORETV, OU)

###################################Nomalize ecDNA 20210303################
def Start41():
    THEORETV = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/Colon-P2_PC3-P2_PDAC_theoretical_value.txt'
    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210303/CellFit/'
    CellFit4().CMD41(IN, INF, THEORETV, OU)
#Start41()

###################################Nomalize ecDNA 20210405################
def Start42():
    THEORETV = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/Colon-P2_PC3-P2_PDAC_theoretical_value.txt'
    IN=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',
        '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',
        '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/EcDNA/20210405/04.EcRegion/All.circle.region.UpFilterTRF',]
    INF='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210405/CellFit/'
    CellFit4().CMD41(IN, INF, THEORETV, OU)
#Start42()

###################################Nomalize ecDNA################
class NomalCounts:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.CHRK=[str(i) for i in range(1,23)] + ['X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def bamfilter(self, inbam):
        Counts  = 0
        samfile = pysam.AlignmentFile(inbam, "rb")
        def filter_read(read):            
            return ((read.flag in [0, 1, 16]) & 
                    (read.reference_name in self.CHRS))

        for read in samfile.fetch():
            if filter_read(read):
                Counts += 1
        samfile.close()
        return Counts

    def _mapreadA(self, inbam):
        f = open(inbam + '.st.stat', 'r')
        mapreads = [i for i in f.readlines() if 'reads mapped' in i]
        f.close()
        if mapreads:
            return int(mapreads[0].split('\t')[-1])
        else:
            #raise ValueError('cannot find the reads mapped line in the file: %s'%Bstat)
            print('cannot find the reads mapped line in the file: %s'%Bstat)
            return 0

    def _mapreads(self, infile):
        readsdict = {}
        for _, _l in infile.iterrows():
            Fstat= '{bamfile}/*/{sid}/{sid}.sorted.bam'.format(bamfile=_l.DNAWorkpath , sid=_l.DNA)
            Bstat= glob.glob(Fstat)
            if Bstat:
                Bstat = Bstat[0]
            else:
                raise ValueError('cannot find the file: %s'%Fstat)
            readsdict[_l.DNA] = Bstat
        KEYS = list(readsdict.keys())
        VULS = list(readsdict.values())

        with futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor/ProcessPoolExecutor
            CouBase = executor.map(self._mapreadA, VULS)
            CouBase = list(CouBase)
            K = [KEYS, VULS, CouBase]
            dump(K, self.outpre + 'readsdict.pkl')
        return readsdict

    def getcounts(self, dfmrx):
        countdict={}
        for _, _l in dfmrx.iterrows():
            if (not _l.gene_name) or (_l.gene_name=='.'):
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            #_S = dict(zip( _l.support_IDs.split(';'), map( int,_l.support_read_num.split(';')) ))
            _S = dict(zip( _l.SIDs.split(';'), map( int,_l.Supports.split(';')) ))
            for _i in list(zip(_G, _B)):
                #if _i[0] !='.':
                countdict.setdefault(_i, []).append(_S)
        countlist = []
        for k, v in countdict.items():
            genedict ={'gene': k[0], 'gene_biotype': k[1]}
            for _d in v:
                for _id, _count in _d.items():
                    if _id in genedict.keys():
                        genedict[_id] += _count
                    else:
                        genedict[_id] = _count
            genedict = pd.Series(genedict)
            countlist.append( genedict )
        countlist = pd.concat(countlist, axis=1).T
        return countlist
    
    def Pdistribut(self, _indf, out, X='Mean', Dup=[], Bins=[], logx=False, logy=False, title='', xlim=[1,100]):
        if not _indf.empty:
            indef = _indf.copy()
            indef = indef[~(indef.gene.isin(self.PLMD))]
            indef['Sum']  = indef.iloc[:,2:].sum(1)
            indef['Mean'] = indef.iloc[:,2:].mean(axis=1, skipna=True)
            K= indef[(indef['gene_biotype'].isin(['protein_coding', 'lncRNA']))]
            print(K, K[(K.Sum>10)].shape, K[(K.Mean>10)].shape)
            if Dup:
                indef = indef[Dup +[X]].drop_duplicates(keep='first')
            indef[X] = indef[X].astype(float)
            if xlim:
                indef = indef[((indef[X]>=xlim[0]) & (indef[X]<=xlim[1])) ]

            if Bins:
                dp = sns.displot(data=indef, x=X, bins=Bins, kde=False, log_scale=logx)     
            else:
                dp = sns.displot(data=indef, x=X, kde=True, log_scale=logx)
            dp.set_xticklabels(rotation=270)

            if title:
                plt.title(title)
            if logy:
                plt.yscale('log')

            plt.tight_layout()
            plt.savefig( out )
            plt.close()

    def ecRegion(self, INs):
        BCounts = pd.DataFrame(columns=['gene', 'gene_biotype'])
        for _, _l in INs.iterrows():
            IN=_l.INF
            CL=_l.Cellline
            UpFilter=pd.read_csv(IN, sep='\t')

            UpFilter = UpFilter[(UpFilter.Type<=7)]
            CHRKD = UpFilter.loc[~(UpFilter['#chrom'].isin(self.CHRK)), 'LINKS'].unique()
            UpChr = UpFilter[~(UpFilter.LINKS.isin(CHRKD))].copy()

            PLMDL = UpFilter.loc[(UpFilter['#chrom'].isin(self.PLMD)), 'LINKS'].unique()
            UpPLD = UpFilter[(UpFilter.LINKS.isin(PLMDL))].copy()
            UpPLD = UpPLD.loc[ (UpPLD.groupby(by='LINKS')['length'].idxmax()) ] # Plasmid only keep the Type='maxlen'
            UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), 'gene_name'] = UpPLD.loc[(UpPLD['#chrom'].isin(self.PLMD)), '#chrom']
            UpFilter= pd.concat([UpChr, UpPLD], axis=0)
            UpFilter[['gene_name', 'gene_biotype']] = UpFilter[['gene_name', 'gene_biotype']].astype(str).fillna('.')

            UpFilter= self.getcounts(UpFilter)
            self.Pdistribut(UpFilter, '%s%s_gene.counts.pdf'%(self.outpre, CL),title = CL)
            BCounts = BCounts.merge(UpFilter, on=['gene', 'gene_biotype'], how='outer')
        BCounts.to_csv(self.outpre + 'gene.counts.txt', sep='\t', index=False)
        return BCounts

    def _getdb(self, INf):
        infile = pd.read_csv(INf, sep='\t')
        infile = infile[(infile.Filter == 'Keep')]
        #self._mapreads(infile)
        MCounts =  pd.DataFrame(load(self.RD),index=['SID', 'INbam', 'Counts']).T
        MCounts.to_csv(self.outpre + 'readsdict.xls', sep='\t', index=False)

        Tail = '/EcDNA/%s/04.EcRegion/All.circle.region.UpFilterTRF'%self.DT
        infile['INF'] = infile.DNAWorkpath + Tail
        INFs = infile[['Cellline', 'INF']].drop_duplicates(keep='first')
        BCounts= self.ecRegion(INFs)

        BCounts = pd.read_csv(self.outpre + 'gene.counts.txt', sep='\t')
        MCounts = dict(zip(MCounts['SID'], MCounts['Counts']))

        BCounts = BCounts.apply(lambda x: x*10e6/MCounts[x.name] if x.name in MCounts else x, axis=0)
        BCounts.to_csv(self.outpre + 'gene.counts.RPM.txt', sep='\t', index=False)

    def _getinfo(self, INf, OUT, Head, DT, RD):
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        self.DT = DT
        self.RD = RD
        os.makedirs(self.outdir, exist_ok=True)

        infile = pd.read_csv(INf, sep='\t')
        Tail = '/EcDNA/%s/04.EcRegion/All.circle.region.UpFilterTRF'%self.DT
        infile['INF'] = infile.DNAWorkpath + Tail
        self.infile = infile[(infile.Filter == 'Keep')]
        return self

    def _getgbed(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['X','Y']
        self.keepbio= ['protein_coding', 'miRNA', 'lncRNA']
        #self.keepbio= ['protein_coding']
        self.genbed = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
        self.cgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.cgc.723.20210225.txt'
        self.sgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.samaticmu.247.txt'
        self.ogene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.OncoKB.org.1064.20210225.txt'

        bgene = pd.read_csv(self.genbed, sep='\t')
        bgene = bgene.loc[((bgene["#chrom"].isin(self.CHRS)) & (bgene.gene_biotype.isin(self.keepbio)))]
        #bgene = bgene.loc[(bgene.gene_biotype.isin(self.keepbio)), ['#chrom', 'start', 'end', 'gene_name']]

        cgene = pd.read_csv(self.cgene, sep='\t').fillna('')
        cgene = cgene[(cgene['Tier'] ==1)]
        clist = cgene['Gene Symbol'].tolist()

        self.bgene = bgene #[~(bgene.gene_name.isin(clist))]
        self.cgene = bgene[(bgene.gene_name.isin(clist))]
        return self
    
    def getcn(self, CN, name='oncogene'):
        return CN.drop(['gene', 'gene_biotype'],axis=1)\
                .sum(0).rename(name).rename_axis(index='DNA') #.reset_index()

    def oncocom(self, Type='.RPM', rep=1000):
        self._getgbed()
        GCounts = pd.read_csv('%sgene.counts%s.txt'%(self.outpre, Type), sep='\t')
        GCounts = GCounts[ (GCounts.gene_biotype.isin(self.keepbio))]

        BCN = GCounts[(GCounts.gene.isin(self.cgene.gene_name.tolist()))]
        BCN = self.getcn(BCN)
        CCN = []
        for i in range(rep):
            idx = np.random.choice(np.arange(self.bgene.shape[0]), size=self.cgene.shape[0], replace=False)
            cge = self.bgene.iloc[idx]
            ccn = GCounts[(GCounts.gene.isin(cge.gene_name.tolist()))]
            CCN.append(self.getcn(ccn, i))
        CCN = pd.concat(CCN, axis=1).mean(1).rename('genome')
        
        ACN = pd.concat([BCN, CCN], axis=1)\
                .reset_index()\
                .merge(self.infile[['DNA', 'Cellline']], on='DNA', how='inner')
        ACN.to_csv('%soncovscall.counts%s.txt'%(self.outpre, Type), sep='\t', index=False)
        cmd = 'Rscript /share/home/share/Pipeline/14EcDNA/EcDNAFinder/Script/GGpaired.R {out}oncovscall.counts{Type}.txt {out}oncovscall.counts{Type}.pdf'.format(out=self.outpre, Type=Type)
        os.system(cmd)

    def getlinks(self, dfmrx):
        countlist=[]
        for _, _l in dfmrx.iterrows():
            if (not _l.gene_name) or (_l.gene_name=='.'):
                continue
            _G = _l.gene_name.split(';')
            _B = _l.gene_biotype.split(';')
            _S = _l.SIDs.split(';')
            _L = _l.LINKS
            for _i in list(zip(_G, _B)):
                if _i[0] !='.':
                    for _s in _S:
                        countlist.append( list(_i) + [_s, _L])
        countlist = pd.DataFrame(countlist, columns=['gene_name', 'gene_biotype', 'DNA', 'LINKS'])
        return countlist

    def _gettnum(self, genedf, geneLink, name='oncogene'):
        geneset = geneLink.merge(genedf, on=['gene_name', 'gene_biotype'], how='inner')
        return geneset.groupby('DNA')['LINKS'].apply(lambda x: len(x.unique())).to_frame(name)

    def _getCir(self, rep=1000):
        INFs = self.infile[['Cellline', 'INF']].drop_duplicates(keep='first')
        ACNs = []
        for _, _l in INFs.iterrows():
            Links = pd.read_csv(_l.INF, sep='\t')
            Drop  = Links.loc[ ~(Links['#chrom'].isin(self.CHRK)), 'LINKS' ]
            Links = Links[~(Links.LINKS.isin(Drop)) ]
            geneLink = self.getlinks(Links)

            BCN = self._gettnum(self.cgene, geneLink)
            CCN = []
            for i in range(rep):
                idx = np.random.choice(np.arange(self.bgene.shape[0]), size=self.cgene.shape[0], replace=False)
                cge = self.bgene.iloc[idx]
                CCN.append(self._gettnum(cge, geneLink, name=i))
            CCN = pd.concat(CCN, axis=1).mean(1).rename('genome')
            ACN = pd.concat([BCN, CCN], axis=1).reset_index()
            ACN['Cellline'] = _l.Cellline
            ACNs.append(ACN)
        ACNs = pd.concat(ACNs, axis=0)
        ACNs.to_csv('%soncovscall.bpnum.txt'%(self.outpre), sep='\t', index=False)
        cmd = 'Rscript /share/home/share/Pipeline/14EcDNA/EcDNAFinder/Script/GGpaired.R {out}oncovscall.bpnum.txt {out}oncovscall.bpnum.pdf'.format(out=self.outpre)
        os.system(cmd)

    def oncobp(self, Type='.RPM', rep=1000):
        self._getgbed()
        self._getCir()

    def CMD(self, INf, OUT, Head, DT, RD):
        self._getinfo(INf, OUT, Head, DT, RD)
        #self._getdb(INf)
        #self.oncocom()
        #self.oncocom(Type='')
        self.oncobp()
        'Rscript /share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/genecounts2searatcounts.R ' + OUT

#samtools  view -@ 20 -bS -F  260  Colon-P1_BC6.sorted.bam | samtools  sort -@ 20 - -o $OU/${ID}.bam 
###################################Nomalize ecDNA 20210225################
def Start51(DT):
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    #DT='20210225'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/GeneCount'%DT
    RD='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/GeneCount/Nomalize.ecgeneCounts.readsdict.pkl'
    Head='Nomalize.ecgeneCounts.'
    NomalCounts().CMD(INfile, OU, Head, DT, RD)

#Start51('20210225')
Start51('20210303')
Start51('20210405')
###################################statistic ecDNA 20210225################
class StateCounts:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def Pdistribut(self, _indf, out, X='query_length', Dup=[], Bins=[], logx=True, logy=True, title='', xlim=None):
        if not _indf.empty:
            indef = _indf.copy()
            if Dup:
                indef = indef[Dup +[X]].drop_duplicates(keep='first')
            indef[X] = indef[X].astype(float)
            if xlim:
                indef = indef[((indef[X]>=xlim[0]) & (indef[X]<=xlim[1])) ]

            if Bins:
                dp = sns.displot(data=indef, x=X, bins=Bins, kde=False, log_scale=logx)     
            else:
                dp = sns.displot(data=indef, x=X, kde=True, log_scale=logx)
            dp.set_xticklabels(rotation=270)

            if title:
                plt.title(title)
            if logy:
                plt.yscale('log')

            plt.tight_layout()
            plt.savefig( out )
            plt.close()

    def ecCounts(self, INs):
        for _, _l in INs.iterrows():
            IN = _l.ecDNApath
            CL = _l.Cellline
            UpFilter=pd.read_csv(IN, sep='\t')
        
            UnChrs = UpFilter.loc[~(UpFilter['#chrom'].isin(self.CHRS)), 'LINKS'].unique()
            UpFilter = UpFilter[~(UpFilter.LINKS.isin(UnChrs))].copy()
            UpFilter = UpFilter[(UpFilter.Supportsum>1)]
            Linksnum = UpFilter[['LINKS','Supportsum']].drop_duplicates(keep='first')
            self.Pdistribut(UpFilter, self.outpre + CL + '.supportnum.log.pdf', 
                                X='Supportsum', Dup=['LINKS'], logx=True, logy=True, title=CL)
            self.Pdistribut(UpFilter, self.outpre + CL + '.supportnum.pdf', 
                                X='Supportsum', Dup=['LINKS'], logx=False, logy=False,
                                Bins = range(1,31), xlim = [0, 30], title=CL)

    def _getinfo(self, INf, OUT, Head, DT ):
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        self.DT = DT
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getdb(self, INf):
        infile = pd.read_csv(INf, sep='\t')
        infile = infile[(infile.Filter == 'Keep')]
        Tail = '/EcDNA/%s/04.EcRegion/All.circle.region.UpMerge'%self.DT
        infile['ecDNApath'] = infile.DNAWorkpath + Tail

        #self._mapreads(infile)
        #MCounts =  pd.DataFrame(load(self.outpre + 'readsdict.pkl'),index=['SID', 'INbam', 'Counts']).T
        #MCounts.to_csv(self.outpre + 'readsdict.xls', sep='\t', index=False)

        INFs = infile[['Cellline', 'ecDNApath']].drop_duplicates(keep='first')
        self.ecCounts(INFs)

        '''
        BCounts = pd.read_csv(self.outpre + 'gene.counts.txt', sep='\t')
        MCounts = dict(zip(MCounts['SID'], MCounts['Counts']))

        BCounts = BCounts.apply(lambda x: x*10e6/MCounts[x.name] if x.name in MCounts else x, axis=0)
        BCounts.to_csv(self.outpre + 'gene.counts.RPM.txt', sep='\t', index=False)
        '''

    def CMD(self, INf, OUT, Head, DT):
        self._getinfo(INf, OUT, Head, DT)
        self._getdb(INf)

def Start52():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210225/EcState'
    DT='20210225'
    RD='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/GeneCount/Nomalize.ecgeneCounts.readsdict.pkl'
    Head='EcState.'
    StateCounts().CMD(INfile, OU, Head, DT)
#Start52()

###################################CNV#######################
class CNV:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getinfo(self, INf, OUT, Head, DT):
        self.infile = pd.read_csv(INf, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.outdir= OUT
        self.DT = DT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getio(self, ftail='.all.cnv.xls'):
        CNVs = []
        for _l in self.infile.DNAWorkpath.unique():
            CNVfs = '{DNAWorkpath}/CNV/{DT}/*{tail}'.format(DNAWorkpath=_l, DT=self.DT, tail=ftail)
            CNVf  = glob.glob(CNVfs)
            if CNVf:
                CNVf = CNVf[0]
            else:
                raise ValueError('cannot find the file: %s'%CNVfs)
            CNVf = pd.read_csv(CNVf, sep='\t')
            CNVs.append(CNVf)
        CNVs = pd.concat(CNVs, axis=0)
        return CNVs

    def ginilike(self, lcnv):
        return ( np.power(2, lcnv)*lcnv ).sum(skipna=True, axis=0)

    def cv(self, lcnv):
        return lcnv.std(skipna=True, axis=0)/lcnv.mean(skipna=True, axis=0)

    def cnvpearson(self, lcnv):
        _lcnv = lcnv.copy().dropna(axis=0, how='any')
        return pd.DataFrame( np.corrcoef(_lcnv, rowvar=False), columns=_lcnv.columns, index=_lcnv.columns)

    def copymatrix(self, vcol='logcopy'):
        logcopy = self.CNVs.pivot(index=['chrom', 'start', 'end', 'length', 'gc', 'rmsk', 'bins' ],
                                    columns=["SID"], values=vcol)

        logcopy.to_csv(self.outpre + 'all.sample.logcnv.txt', sep='\t', index=True)

        Score = pd.concat([self.cv(np.power(2, logcopy)), self.ginilike(logcopy)], axis=1)
        Score.columns = ['CV', 'Gini']
        Score.reset_index(inplace=True)
        Score = self.infile[['SID', 'Rename', 'RNA', 'Group', 'Cellline', 'Platform']]\
                    .merge(Score, on=['SID'], how='right')
        Score.to_csv(self.outpre + 'all.sample.logcnv.cvgini.txt', sep='\t', index=False)
        PLOT(self.outpre + 'all.sample.logcnv.cv.pdf').Cnvbox(Score)
        PLOT(self.outpre + 'all.sample.logcnv.gini.pdf').Cnvbox(Score, y='Gini')

        Cor  = self.cnvpearson(logcopy)
        Cor.to_csv(self.outpre + 'all.sample.logcnv.pearson.txt', sep='\t', index=True)
        PLOT(self.outpre + 'all.sample.logcnv.pearson.pdf')\
            .ClustMap(Cor, self.infile[['SID', 'Platform', 'Cellline']].set_index('SID') )

    def CMD(self, INfile, OU, Head, DT):
        self._getinfo(INfile, OU, Head, DT)
        self.CNVs = self._getio()
        self.CNVs = self.CNVs[(self.CNVs.SID.isin(self.infile.SID))]
        self.CNVs.to_csv(self.outpre + 'all.sample.txt', sep='\t', index=False)
        self.copymatrix()

def Start6():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/CNV'
    Head='CNV.Analyze.'
    CNV().CMD(INfile, OU, Head)

def Start61():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210225/CNV'
    Head='CNV.Analyze.'
    DT='20210121'
    CNV().CMD(INfile, OU, Head, DT)

###################################RNA#######################
class RNA():
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']

    def _getinfo(self, INf, OUT, Head ):
        self.infile = pd.read_csv(INf, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.genbed = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
        self.genbed = pd.read_csv(self.genbed, sep='\t')[['#chrom', 'start', 'end', 'gene_name', 'gene_id', 'gene_biotype']]\
                        .rename(columns={'gene_name' : 'gene', 'gene_id' : 'gene_ID'}).copy()
        self.outdir= OUT
        self.Head  = Head
        self.outpre= OUT + '/' + Head
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getio(self, ftail='.rsemce.genes.results'):
        RNAs = []
        for _, _l in self.infile.iterrows():
            RNVfs= '{RNA}/{RNAID}/SS2/RSEM/{RNAID}*{tail}'.format(RNA=_l.RNAWorkpath , RNAID=_l.RNAID, tail=ftail)
            RNVf = glob.glob(RNVfs)
            if RNVf:
                RNVf = RNVf[0]
            else:
                raise ValueError('cannot find the file: %s'%RNVf)
            RNVf = pd.read_csv(RNVf, sep='\t')
            RNVf.insert(0, 'RNAID', _l.RNAID)
            RNVf[['gene_ID','gene']] = RNVf['gene_id'].str.split('_', expand=True)
            RNAs.append(RNVf[['RNAID', 'gene', 'gene_ID', 'TPM', 'FPKM']])
        RNAs = pd.concat(RNAs, axis=0)
        return RNAs

    def pearsonMT(self, rdf):
        rdf = rdf[ (rdf.sum(1)>2) ].copy()
        return pd.DataFrame( np.corrcoef(rdf, rowvar=False), columns=rdf.columns, index=rdf.columns)

    def expreseq(self, eqtype='TPM'):
        self.EQ = self.RNAs.pivot(index=['gene', 'gene_ID'], columns='RNA', values=eqtype)
        self.EQ.to_csv( '%sall.sample.%s.txt'%(self.outpre, eqtype), sep='\t', index=True)

    def RNAmtr(self):
        self.RNAs = self._getio()
        self.RNAs = self.infile[['Rename','RNA', 'RNA', 'Group', 'Cellline', 'Platform', 'RNAID']]\
                        .merge(self.RNAs, on='RNAID', how='right')
        self.RNAs.to_csv(self.outpre + 'all.sample.txt', sep='\t', index=False)

    def meanEQ(self, RNAs, eqtype='TPM'):
        for (_p, _c), _g in RNAs.groupby(by=['Platform', 'Cellline']):
            _G = _g.groupby(by=['gene_ID', 'gene'])[eqtype].mean()\
                    .to_frame('mean'+eqtype).reset_index()\
                    .merge(self.genbed, on=['gene_ID', 'gene'], how='left')
            _G = _G[['#chrom', 'start', 'end', 'gene_ID', 'gene', 'gene_biotype', 'mean'+eqtype]]\
                    .sort_values(by=['#chrom', 'start', 'end'])
            _G = _G[((_G['#chrom'].isin(self.CHRS)) & (_G['gene_biotype']=='protein_coding'))]
            _G.to_csv('%sall.sample.mean%s_%s_%s.txt'%(self.outpre, eqtype, _p, _c), sep='\t', index=False)

            #_K = _G.loc[(_G.meanTPM>0), ['#chrom', 'start', 'end','meanTPM']].copy()
            #_K['meanTPM'] = np.log2(_K['meanTPM'])
            #_K['#chrom']  = 'hs' + _K['#chrom']
            #_K.to_csv('%sall.sample.logmean%s_%s_%s.txt'%(self.outpre, eqtype, _p, _c), sep='\t', index=False)

    def CMD(self, INfile, OU, Head):
        self._getinfo(INfile, OU, Head)
        self.infile = self.infile[(~self.infile.RNAID.isna())]

        self.RNAmtr()
        self.RNAs = pd.read_csv(self.outpre + 'all.sample.txt', sep='\t')
        self.meanEQ(self.RNAs)
        self.meanEQ(self.RNAs, eqtype='FPKM')
        self.expreseq()
        self.expreseq(eqtype='FPKM')

        self.Cor  = self.pearsonMT(self.EQ)
        self.Cor.to_csv(self.outpre + 'all.sample.pearson.txt', sep='\t', index=True)
        PLOT(self.outpre + 'all.sample.pearson.pdf')\
            .ClustMap(self.Cor, self.infile[['RNA', 'Platform', 'Cellline']].set_index('RNA'))

def Start7():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210128.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210131/RNA'
    Head='RNA.Analyze.'
    RNA().CMD(INfile, OU, Head)

def Start71():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210225/RNA'
    Head='RNA.Analyze.'
    RNA().CMD(INfile, OU, Head)

#Start71()
###############################cellline circos###########
class CCircos:
    def __init__(self, filt=False, RNAfilt=20):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.CHRK=[str(i) for i in range(1,23)] + ['X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.filter  = filt
        self.RNAfilt = RNAfilt
        self.keepbio= ['protein_coding', 'miRNA', 'lncRNA']
        self.genbed = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
        self.cgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.cgc.723.20210225.txt'
        self.sgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.samaticmu.247.txt'
        self.ogene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.OncoKB.org.1064.20210225.txt'

    def _filtgene(self):
        bgene = pd.read_csv(self.genbed, sep='\t')
        bgene['#chrom'] = 'hs' + bgene['#chrom'].astype(str).str.lstrip('chr')
        bgene = bgene[['#chrom', 'start', 'end', 'gene_name']]
        #bgene = bgene.loc[(bgene.gene_biotype.isin(self.keepbio)), ['#chrom', 'start', 'end', 'gene_name']]
        if self.filter:
            cgene = pd.read_csv(self.cgene, sep='\t').fillna('')
            #cgene = cgene[(cgene['Role in Cancer'].str.contains('oncogene'))]
            cgene = cgene[(cgene['Tier'] ==1)]
            #cgene = pd.read_csv(self.ogene, sep='\t')
            #cgene = cgene[(cgene['Is Oncogene']=='Yes')]
            Kgene = cgene.iloc[:,0]
            bgene = bgene[(bgene.gene_name.isin(Kgene))]
        return bgene

    def geneannot(self, Links, Gene):
        targ= set(';'.join(Links.gene_name.fillna('.')).split(';')) & set( Gene.gene_name.tolist())
        pd.DataFrame(list(targ)).to_csv(self.GOKEGG+'/cancer.gene.txt', sep='\t', index=False, header=False)

        Genes = self.RNAex[ (self.RNAex.gene.isin(targ))]\
                    .sort_values(by='meanFPKM',ascending=False)\
                    .head(self.RNAfilt)
        return Genes[['#chrom', 'start', 'end', 'gene']]

    def _getio(self, INfile, WD, HD, DT, PL):
        self.infile = pd.read_csv(INfile, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.WD=WD
        self.HD=HD
        self.DT=DT
        self.PL=PL
        self.WP = '%s/%s'%( WD, HD)
        self.OUT='%s/%s/Circos/%s/'%( WD, HD, DT)
        self.OUTdt= self.OUT+'/data/'
        self.GOKEGG = '%s/%s/GoKegg/%s/'%( WD, HD, DT)
        os.makedirs(self.OUTdt, exist_ok=True)
        os.makedirs(self.GOKEGG, exist_ok=True)
        return self

    def _getECdt(self, Dt='20210303'):
        ecPa  = '%s/EcDNA/%s/04.EcRegion/data/'%(self.WP, Dt)
        os.system('ln -sf %s/links.txt %s'%(ecPa, self.OUTdt))
        os.system('ln -sf %s/links.num.txt %s'%(ecPa, self.OUTdt))
        os.system('rm  %s/links.gene.txt'%(self.OUTdt))

        Links = pd.read_csv('%s/EcDNA/%s/04.EcRegion/All.circle.region.UpFilterTRF'%(self.WP, Dt), sep='\t')
        Drop  = Links.loc[ ~(Links['#chrom'].isin(self.CHRK)), 'LINKS' ]
        Links = Links[ ~(Links.LINKS.isin(Drop)) ]
        Links['#chrom'] = 'hs' + Links['#chrom'].astype(str).str.lstrip('chr')
    
        genel = self.geneannot(Links, self._filtgene())
        genel.to_csv(self.OUTdt+'/links.gene.txt', sep='\t', index=False, header=False)

    def _getCNVdt(self, Dt='20210121'):
        if self.HD == 'PDAC':
            Dt = '20210405'
            cnvPa = '%s/CNV/%s/%s.Case.cnv.logmean.xls'%(self.WP, Dt, self.HD)
        else:
            cnvPa = '%s/CNV/%s/%s.all.cnv.logmean.xls'%(self.WP, Dt, self.HD)
        CNVmean = pd.read_csv(cnvPa, sep='\t')
        CNVmean = CNVmean[['chrom', 'start', 'end', 'logcopy', 'meanlog2CN']]
        CNVmean = CNVmean[(CNVmean.logcopy >=-5)]
        CNVmean['chrom'] = ('hs' + CNVmean['chrom'].astype(str)).replace({'hs23':'hsX','hs24':'hsY'})

        CNVmean[['chrom', 'start', 'end', 'logcopy']].to_csv(self.OUTdt +'cnv.logcopy.txt', sep='\t', index=False, header=False)
        CNVmean[['chrom', 'start', 'end', 'meanlog2CN']].to_csv(self.OUTdt +'cnv.meanlogcopy.txt', sep='\t', index=False, header=False)

    def _getRNAdt(self, Dt='20210225', Exn='FPKM'):
        rnaPa = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/RNA/RNA.Analyze.all.sample.mean%s_%s_%s.txt'\
                    %(Dt, Exn, self.PL, self.HD)
        COL = ['#chrom', 'start', 'end', 'mean'+Exn]
        RNAex = pd.read_csv(rnaPa, sep='\t')
        RNAex = RNAex[(RNAex['mean'+Exn] >0)]
        RNAex['#chrom'] = 'hs' + RNAex['#chrom'].astype(str)
        RNAex[COL].to_csv(self.OUTdt +'rna.txt', sep='\t', index=False, header=False)
        self.RNAgene =  RNAex['gene'].tolist()
        self.RNAex = RNAex
        return self

    def goplot(self):
        Conf={'out': self.OUT, 
                'Perl':'/share/home/share/software/Perl-5.32.1/bin/perl', 
                'Circos':'/share/home/share/software/circos-0.69-9/bin/circos', 
                'Circnf':'/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Circos/circos.dna.rna.conf',
                'legend': '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Circos/legend.part.svg',
                'HD': self.HD}
        cmd='cd {out} ; {Perl} {Circos} -conf {Circnf} -outputdir {out} -outputfile {HD}.link.site.svg'\
                .format(**Conf)
        legend= "sed -i '/<defs>/ e cat %s' %s/%s.link.site.svg"%(Conf['legend'], self.OUT, self.HD)
        os.system(cmd)
        os.system(legend)
        try:
            import cairosvg
            ohead = '%s/%s.link.site'%(self.OUT, self.HD)
            cairosvg.svg2pdf(url=ohead+'.svg', write_to=ohead+'.pdf')
        except:
            pass

    def _GoKegg(self):
        Conf ={'Rscript': '/share/home/share/software/R-3.6.3/bin/Rscript',
                'GOKEGG': '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Script/GOKEGG.R',
                'INfile': self.GOKEGG+'/cancer.gene.txt',
                'Outdir': self.GOKEGG}
        cmd = '{Rscript} {GOKEGG} {INfile} {Outdir}'.format(**Conf) 
        os.system(cmd)
        #cmd = 'Rscript /share/home/share/Pipeline/14EcDNA/EcDNAFinder/Script/GGpaired.R {out}oncovscall.counts{Type}.txt {out}oncovscall.counts{Type}.pdf'.format(out=self.outpre, Type=Type)
        #os.system(cmd)

    def CMD(self, INfile, WD, HD, DT, PL):
        self._getio(INfile, WD, HD, DT, PL)
        self._getCNVdt()
        self._getRNAdt()
        self._getECdt(Dt=DT)
        self.goplot()
        self._GoKegg()
def Start81(WD, HD, PL, DT, Filt=False, RNAfilt=20 ):
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    CCircos(filt=Filt, RNAfilt=RNAfilt).CMD(INfile, WD, HD, DT, PL)
#Start81('/data/zhouwei/01Projects/03ecDNA/PacbioCCS/','U2OS','Pacbio','20210303' )
#Start81('/data/zhouwei/01Projects/03ecDNA/PacbioCCS/','HEK293T','Pacbio','20210303' )
#Start81('/data/zhouwei/01Projects/03ecDNA/Nanopore/','U2OS','Nanopore','20210303' )
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/','COLON','Nanopore','20210303' )
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/','PC3','Nanopore','20210303' )
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore','PDAC','Nanopore','20210303' )

#Start81('/data/zhouwei/01Projects/03ecDNA/PacbioCCS/','U2OS','Pacbio','20210405', Filt=True, RNAfilt=20)
#Start81('/data/zhouwei/01Projects/03ecDNA/PacbioCCS/','HEK293T','Pacbio','20210405', Filt=True, RNAfilt=20 )
#Start81('/data/zhouwei/01Projects/03ecDNA/Nanopore/','U2OS','Nanopore','20210405', Filt=True, RNAfilt=20)
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/','COLON','Nanopore','20210405', Filt=True, RNAfilt=20 )
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/','PC3','Nanopore','20210405', Filt=True, RNAfilt=20 )
#Start81('/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore','PDAC','Nanopore','20210405', Filt=True, RNAfilt=20 )

###############################bam state################
class Fig1s:
    def __init__(self):
        self.color_ = [ '#00DE9F', '#FF33CC', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]
    def _getinfo(self, INfile, Bstate, OU, HD):
        self.infile = pd.read_csv(INfile, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'ID'})
        self.Bstate = Bstate
        self.OUT    = OU
        self.OUTpre = OU + '/' + HD
        os.makedirs(self.OUT, exist_ok=True)
        return self

    def _getstate(self):
        Bstate = []
        for i in self.Bstate:
            bn = os.path.basename(i)
            pl, cl = bn.split('.')[0:2]
            pl = pl.capitalize()
            bs = pd.read_csv(i, sep='\t')
            if cl in ['Colon-P1', 'Colon-P2', 'PC3-P1', 'PC3-P2' ]:
                bs['ID'] = cl + '_' + bs['ID']
            Bstate.append(bs)
        Bstate = pd.concat(Bstate, axis=0, sort=False)
        Bstate.to_csv(self.OUTpre + 'allsample.xls',sep='\t',index=False)

        Bstate = self.infile.merge(Bstate, on='ID', how='left')
        Bstate.to_csv(self.OUTpre + 'allsample.info.xls',sep='\t',index=False)
        return Bstate

    def cellBox(self, xyData, out, x = 'Cellline', y='mapped_reads', ylim=(), scalet='width'):
        RName = ['HEK293T','U2OS.CCS', 'U2OS', 'COLON','PC3']
        NName = ['HEK293T_Pacbio','U2OS_Pacbio', 'U2OS_Nanopore', 'COLON320DM_Nanopore','PC3_Nanopore']

        xyData[x] = xyData[x].replace(dict(zip(RName, NName)))
        xyData = xyData[(xyData[x]!='PDAC')]
        Corder = ['#20b2aa','#699146','#807dba','#6a51a3','#e31a1c','#ff7f00']

        fig, ax = plt.subplots(figsize=(5,5))
        Col1 = sns.set_palette(sns.color_palette(Corder))
        #Col2 = sns.set_palette(sns.color_palette(self.color_[:6]))
        
        #sns.boxplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.45,  order=Sorder,
        #                data=xyData, palette=Col1,  fliersize=3, linewidth=1, ax=ax)

        sns.violinplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.43,  order=NName,
                        inner = 'box', #cut =0,
                        data=xyData, palette=Col1, linewidth=0.8, scale=scalet, ax=ax)
        sns.swarmplot(x=x, y=y, data=xyData,  order=NName, color='black', 
                        #palette=Col1, edgecolor='black', linestyles='-', 
                        size=2.2, linewidth=.1, ax=ax)
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y',useOffset=False)
        #plt.xticks(rotation='45')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(y)
        if ylim:
            plt.ylim(ylim)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( out, bbox_inches='tight')
        plt.close()

    def CMD(self, INfile, Bstate, OU, HD):
        self._getinfo(INfile, Bstate, OU, HD)
        Bstate = self._getstate().sort_values(by=['Platform','Cellline'], ascending=[False, True])

        Keep = ['mapped_reads', 'mapped_reads_rate', 'average_length', 
                    'maximum_length', 'mapped_bases_rate', 'error_rate', 
                    'mean_coverage', 'mean_depth', 'chromreadfreq']
        #Keep = ['mapped_reads']
        for i in Keep:
            if i in ['mapped_reads_rate']:
                self.cellBox(Bstate, self.OUTpre + i + '.pdf', y = i, ylim=[0.9,1.004])
            elif i in ['mean_coverage']:
                self.cellBox(Bstate, self.OUTpre + i + '.pdf', y = i, ylim=[0, 1])
            else:
                self.cellBox(Bstate, self.OUTpre + i + '.pdf', y = i)
def f1s():
    Bstate=['/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/BamStat/nanopore.Colon-P1.MINIMAPont.bam.stat.xls',
            '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/BamStat/nanopore.Colon-P2.MINIMAPont.bam.stat.xls',
            '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/BamStat/nanopore.PC3-P1.MINIMAPont.bam.stat.xls',
            '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PC3/BamStat/nanopore.PC3-P2.MINIMAPont.bam.stat.xls',
            '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/BamStat/nanopore.PDAC1.bam.stat.xls',
            '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/PDAC/BamStat/nanopore.PDAC2.bam.stat.xls',
            '/data/zhouwei/01Projects/03ecDNA/Nanopore/U2OS/BamStat/nanopore.U2OS.bam.stat.xls',
            '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/HEK293T/BamStat/Pacbio.HEK293T.PBMM2.bam.stat.xls',
            '/data/zhouwei/01Projects/03ecDNA/PacbioCCS/U2OS/BamStat/Pacbio.U2OS.PBMM2.bam.stat.xls']
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU  ='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210303/Bamstate'
    DT  = '20210303'
    HD  = 'bam.state.'
    Fig1s().CMD(INfile, Bstate, OU, HD)
#f1s()
class ReadLen:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.CHRK=[str(i) for i in range(1,23)] + ['X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
    
    def _getio(self, OU, HD):
        self.OUT    = OU
        self.OUTpre = OU + '/' + HD
        os.makedirs(self.OUT, exist_ok=True)
        return self

    def _getinfo(self):
        info = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
        self.infile = pd.read_csv(info, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')]

    def _readlen(self, args):
        SID, inbam = args[:2]
        Lens  = {}
        samfile = pysam.AlignmentFile(inbam, "rb")
        def filter_read1(read):            
            return ((read.flag in [0, 1, 16]) & 
                    (read.reference_name in self.CHRS))
        def filter_read(read):            
            return (read.reference_name in self.CHRS)

        for read in samfile.fetch():
            if filter_read(read):
                Lens[read.query_name] = int(read.query_length)
        samfile.close()
        Lenl = list(Lens.values())
        return [SID, sum(Lenl)/len(Lenl), max(Lenl), np.median(Lenl)]
    
    def _mapreads(self, infile):
        readsdict = {}
        for _, _l in infile.iterrows():
            Fstat= '{bamfile}/*/{sid}/{sid}.sorted.bam'.format(bamfile=_l.DNAWorkpath , sid=_l.DNA)
            Bstat= glob.glob(Fstat)
            if Bstat:
                Bstat = Bstat[0]
            else:
                raise ValueError('cannot find the file: %s'%Fstat)
            readsdict[_l.DNA] = Bstat
        VULS = [ (k,v) for k,v in readsdict.items() ]

        with futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor/ProcessPoolExecutor
            CouBase = executor.map(self._readlen, VULS)
            CouBase = list(CouBase)
            pd.DataFrame(CouBase).to_csv(self.OUTpre + 'reads.len.txt', sep='\t', index=False, header=False)
        return readsdict

    def cellBox(self, xyData, out, x = 'Cellline', y='mapped_reads', ylim=(), scalet='width'):
        RName = ['HEK293T','U2OS.CCS', 'U2OS', 'COLON','PC3']
        NName = ['HEK293T_Pacbio','U2OS_Pacbio', 'U2OS_Nanopore', 'COLON320DM_Nanopore','PC3_Nanopore']

        xyData[x] = xyData[x].replace(dict(zip(RName, NName)))
        xyData = xyData[(xyData[x]!='PDAC')]
        Corder = ['#20b2aa','#699146','#807dba','#6a51a3','#e31a1c','#ff7f00']

        fig, ax = plt.subplots(figsize=(5,5))
        Col1 = sns.set_palette(sns.color_palette(Corder))
        #Col2 = sns.set_palette(sns.color_palette(self.color_[:6]))
        
        #sns.boxplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.45,  order=Sorder,
        #                data=xyData, palette=Col1,  fliersize=3, linewidth=1, ax=ax)

        sns.violinplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.43,  order=NName,
                        inner = 'box', #cut =0,
                        data=xyData, palette=Col1, linewidth=0.8, scale=scalet, ax=ax)
        sns.swarmplot(x=x, y=y, data=xyData,  order=NName, color='black', 
                        #palette=Col1, edgecolor='black', linestyles='-', 
                        size=2.2, linewidth=.1, ax=ax)
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y',useOffset=False)
        #plt.xticks(rotation='45')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(y)
        if ylim:
            plt.ylim(ylim)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( out, bbox_inches='tight')
        plt.close()

    def cellBox1(self, xyData, out, x = 'Cellline', y='mapped_reads', ylim=(), scalet='width'):
        RName = ['HEK293T','U2OS.CCS', 'U2OS', 'COLON','PC3']
        NName = ['HEK293T_Pacbio','U2OS_Pacbio', 'U2OS_Nanopore', 'COLON320DM_Nanopore','PC3_Nanopore']

        xyData[x] = xyData[x].replace(dict(zip(RName, NName)))
        xyData = xyData[(xyData[x]!='PDAC')]
        Corder = ['#20b2aa','#699146','#807dba','#6a51a3','#e31a1c','#ff7f00']

        fig, ax = plt.subplots(figsize=(5,5))
        Col1 = sns.set_palette(sns.color_palette(Corder))
        sns.violinplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.43,  order=NName,
                        inner = None, cut=0,  bw='silverman',
                        data=xyData, palette=Col1, linewidth=0.8, scale=scalet, ax=ax)
        for i in ax.collections:
            i.set_edgecolor('grey')
        sns.boxenplot(x=x, y=y, data=xyData, width=0.12, order=NName, ax=ax, color='white', palette=None,
                       linewidth=1, scale='linear', showfliers=False,
                     )
                    #fliersize=3, linewidth=0.8,
                    #notch =True,
                    #meanprops={'linestyle':'-.'},
                    #flierprops={'markersize':3, 'marker':'*', 'linestyle':'--', },
                    #)
        sns.swarmplot(x=x, y=y, data=xyData,  order=NName, color='black', edgecolor='black',
                        #palette=Col1, , linestyles='-', 
                        size=1.5, linewidth=.1, ax=ax)


        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y',useOffset=False)
        #plt.xticks(rotation='45')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(y)
        if ylim:
            plt.ylim(ylim)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( out, bbox_inches='tight')
        plt.close()

    def cellBox(self, xyData, out, x = 'Cellline', y='mapped_reads', ylim=(), scalet='width'):
        RName = ['HEK293T','U2OS.CCS', 'U2OS', 'COLON','PC3']
        NName = ['HEK293T_Pacbio','U2OS_Pacbio', 'U2OS_Nanopore', 'COLON320DM_Nanopore','PC3_Nanopore']

        xyData[x] = xyData[x].replace(dict(zip(RName, NName)))
        xyData = xyData[(xyData[x]!='PDAC')]
        Corder = ['#20b2aa','#699146','#807dba','#6a51a3','#e31a1c','#ff7f00']

        fig, ax = plt.subplots(figsize=(5,5))
        Col1 = sns.set_palette(sns.color_palette(Corder))
        #Col2 = sns.set_palette(sns.color_palette(self.color_[:6]))
        
        #sns.boxplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.45,  order=Sorder,
        #                data=xyData, palette=Col1,  fliersize=3, linewidth=1, ax=ax)

        sns.violinplot(x=x, y=y,  meanprops={'linestyle':'-.'}, width=0.43,  order=NName,
                        inner = 'box', #cut =0,
                        data=xyData, palette=Col1, linewidth=0.8, scale=scalet, ax=ax)
        sns.swarmplot(x=x, y=y, data=xyData,  order=NName, color='black', 
                        #palette=Col1, edgecolor='black', linestyles='-', 
                        size=2.2, linewidth=.1, ax=ax)
        plt.ticklabel_format(style='sci',scilimits=(0,0), axis='y',useOffset=False)
        #plt.xticks(rotation='45')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title(y)
        if ylim:
            plt.ylim(ylim)
        #leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), numpoints=1)
        #plt.savefig( self.out, bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.savefig( out, bbox_inches='tight')
        plt.close()

    def CMD(self, OU, HD):
        self._getio(OU, HD)
        self._getinfo()
        #self._mapreads(self.infile)

        readlen = pd.read_csv(self.OUTpre + 'reads.len.txt', sep='\t', 
                                header=None, names=['DNA', 'average_length_A', 'maximum_length_A', 'median_length_A' ])\
                    .merge(self.infile, on='DNA', how='left')
        for i in ['average_length_A', 'maximum_length_A', 'median_length_A']:
            self.cellBox(readlen, self.OUTpre + i + '.pdf', y = i)
        
        print(readlen)

def Start111():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    OU  ='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210303/Bamstate'
    DT  = '20210303'
    HD  = 'bam.state.'
    ReadLen().CMD(OU, HD)

#Start111()
###############################single link circos########
class SCircos:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.CHRK=[str(i) for i in range(1,23)] + ['X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.genbed = '/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed'
        self.Cole= ['50,150,255', '255,255,0', '0,255,0', '0,255,255',  '255,0,0',  '220,220,220']
        self.Cole2= ['50,150,255', '255,255,0', '0,255,0', '0,255,255',  '255,0,0',  '120,120,120']
        self.color_ = [ '0,222,159', '255,51,204','231,167,45', '139,0,204', '0,114,178', '22,230,255', 
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]
        self.colol_ = [ '#00DE9F', '#FF33CC', '#16E6FF', '#E7A72D', '#8B00CC', '#EE7AE9',
                        '#B2DF8A', '#CAB2D6', '#B97B3D', '#0072B2', '#FFCC00', '#0000FF',
                        '#FF2121', '#8E8E38', '#6187C4', '#FDBF6F', '#666666', '#33A02C',
                        '#FB9A99', '#D9D9D9', '#FF7F00', '#1F78B4', '#FFFFB3', '#5DADE2',
                        '#95A5A6', '#FCCDE5', '#FB8072', '#B3DE69', '#F0ECD7', '#CC66CC',
                        '#A473AE', '#FF0000', '#EE7777', '#009E73', '#ED5401', '#CC0073',]
        self.Cole1= ['white', 'lgrey']
        self.cgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.cgc.723.20210225.txt'
        self.sgene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.samaticmu.247.txt'
        self.ogene  = '/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.OncoKB.org.1064.20210225.txt'

    def _filtgene(self):
        bgene = pd.read_csv(self.genbed, sep='\t')
        bgene['#chrom'] = 'hs' + bgene['#chrom'].astype(str).str.lstrip('chr')
        bgene = bgene[['#chrom', 'start', 'end', 'gene_name']]
        #bgene = bgene.loc[(bgene.gene_biotype.isin(self.keepbio)), ['#chrom', 'start', 'end', 'gene_name']]
        if self.filter:
            cgene = pd.read_csv(self.cgene, sep='\t').fillna('')
            #cgene = cgene[(cgene['Role in Cancer'].str.contains('oncogene'))]
            cgene = cgene[(cgene['Tier'] ==1)]
            #cgene = pd.read_csv(self.ogene, sep='\t')
            #cgene = cgene[(cgene['Is Oncogene']=='Yes')]
            Kgene = cgene.iloc[:,0]
            bgene = bgene[(bgene.gene_name.isin(Kgene))]
        return bgene

    def _getgene1(self, _link, gene_bt=['protein_coding'], minover=30):
        _chr, _start, _end = _link[0], min(_link[1:]), max(_link[1:])
        set0 = (self.genbed['#chrom'] == _chr) & (self.genbed['gene_biotype'].isin(gene_bt)) 
        set1 = (self.genbed.start>= _start) & (self.genbed.end<= _end) 
        set2 = (self.genbed.start< _start)  & (self.genbed.end>= _start+minover)
        set3 = (self.genbed.start<= _end-minover) & (self.genbed.end> _end)
        targetg = self.genbed[ (set0 & (set1 | set2 | set3)) ].copy()
        if _link[1:3] == sorted(_link[1:3]):
            targetg.sort_values(by=['#chrom', 'start', 'end'], ascending=[True, True,  True], inplace=True)
        else:
            targetg.sort_values(by=['#chrom', 'start', 'end'], ascending=[True, False,False], inplace=True)
        return targetg

    def _karyotype(self, linkbed):
        Cola= self.Cole*linkbed.shape[0]
        Chr = linkbed.copy()
        Chr['#chr'] = 'chr'
        Chr['stub'] = '-'
        Chr['ID']   = Chr.gene
        Chr['LABEL'] = Chr.gene
        Chr['START'] = 0
        Chr['END']   = Chr.end - Chr.start
        Chr['COLOR'] = Cola[:Chr.shape[0]-1] + Cola[1:2] \
                            if (Chr.shape[0] % len(self.Cole) ==1) and (Chr.shape[0] >1) \
                            else Cola[:Chr.shape[0]]
        Chr.iloc[:,-7:].to_csv(self.outpre + 'karyotype.txt', sep=' ', index=False)

    def _getkaryo(self):
        linkarr = pd.concat( [self._getgene(i) for i in self.Links], axis=0, sort=False)
        linkarr.to_csv(self.outpre + 'gene.bed', sep='\t')
        self._karyotype(linkarr)

    def _karyotype11(self, linkbed):
        Chr = [['chr', '-', 'Links', 'Links', 0, self.LinksLen, 'white']]
        Chr = pd.DataFrame(Chr, columns=['#chr', 'stub', 'ID', 'LABEL', 'START', 'END', 'COLOR'])

        Cola= self.Cole*linkbed.shape[0]
        Ban = linkbed.copy()
        Ban['#chr'] = 'band'
        Ban['stub'] = 'Links'
        Ban['ID']   = Ban.gene
        Ban['LABEL'] = Ban.gene
        Ban['START'] = Ban[['start', 'end']].min(1)
        Ban['END']   = Ban[['start', 'end']].max(1)
        Ban['COLOR'] = Cola[:Ban.shape[0]-1] + Cola[1:2] \
                            if (Ban.shape[0] % len(self.Cole) ==1) and (Ban.shape[0] >1) \
                            else Cola[:Ban.shape[0]]
        Chr = pd.concat([Chr, Ban], join='inner', axis=0)
        Chr.to_csv(self.outpre + 'karyotype.trans.txt', sep=' ', index=False)

    def _karyotype1(self, linkbed):
        Cola= self.Cole1*len(self.LinksNew)
        Chr = [['chr', '-', 'Links', 'Links', 0, self.LinksLen, 'white']] + \
                [['band', 'Links', i[0], i[0], min(i[1:3]), max(i[1:3]), Cola[n]] for n,i in enumerate(self.LinksNew)]
        Chr = pd.DataFrame(Chr, columns=['#chr', 'stub', 'ID', 'LABEL', 'START', 'END', 'COLOR'])
        Chr.to_csv(self.outpre + 'karyotype.trans.txt', sep=' ', index=False)

    def _highlights1(self, _link):
        _hl = _link.iloc[:,:4].merge(self.RNA, on='gene', how='left')

        _hl['gene'] = _hl['gene'] + '_' +_hl['#chrom']
        _hl['#chrom'] = 'Links'
        _hl[['#chrom', 'start', 'end', 'gene']].to_csv(self.outpre + 'text.trans.txt', sep=' ', index=False)

        _hl[['#chrom', 'start', 'end', 'TPM']].to_csv(self.outpre + 'rna.trans.txt', sep=' ', index=False)

        Cola= [ 'fill_color=' +i for i in (self.Cole2*_hl.shape[0])[:_hl.shape[0]] ]
        _hl['gene'] = Cola 
        _hl[['#chrom', 'start', 'end', 'gene']].to_csv(self.outpre + 'highlights.trans.txt', sep=' ', index=False)

    def _changesite(self, _ilink, _rawL, _newL):
        _ilink = _ilink.copy()
        K = _ilink[['start', 'end']].values
        K[ K >max(_rawL[1:])] = max(_rawL[1:])
        K[ K <min(_rawL[1:])] = min(_rawL[1:])
        _ilink[['start', 'end']] = K
        _ilink['#chrom'] = _ilink['#chrom'].astype(str)

        if sorted(_rawL[1:]) == _rawL[1:]:
            _ilink[['start', 'end']] = _ilink[['start', 'end']]  - _rawL[1] + _newL[1]
        else:
            _ilink[['start', 'end']]  = _rawL[1] - _ilink[['start', 'end']] + _newL[1]
            _ilink['#chrom'] = 'r' + _ilink['#chrom']
        return _ilink

    def _getkaryo1(self):
        BR, BN = [], []
        for _i, _r in enumerate(self.Links):
            _n = self.LinksNew[_i]
            _TR = self._getgene(_r)
            _TN = self._changesite( _TR, _r, _n)
            BR.append(_TR)
            BN.append(_TN)
        BR = pd.concat(BR, axis=0)
        BN = pd.concat(BN, axis=0)

        BR.to_csv(self.outpre + 'raw.gene.site.txt', sep='\t', index=False)
        BN.to_csv(self.outpre + 'new.gene.site.txt', sep='\t', index=False)
        self._karyotype1(BN)
        self._highlights1(BN)

    def _geteccv(self):
        EcCV = self.OUT + '/GeneCount/Nomalize.ecgeneCounts.gene.counts.RPM.txt'
        EcCV = pd.read_csv(EcCV, sep='\t')
        print(EcCV)
    
    def _getrna(self, celli =['U2OS']):
        RNA = self.OUT + '/RNA/RNA.Analyze.all.sample.txt'
        RNA = pd.read_csv(RNA, sep='\t')
        RNA = RNA[(RNA.Cellline.isin(celli))].copy()
        self.RNA = RNA.groupby('gene')['TPM'].mean().reset_index()
        return self

    def _getinfo(self, INf, OUT, Head, LINKS):
        self.infile = pd.read_csv(INf, sep='\t')
        self.infile = self.infile[(self.infile.Filter == 'Keep')].rename(columns={'DNA': 'SID'})
        self.outdir = OUT
        self.outdt  = self.outdir + '/data/'

        self.LINKS = pd.read_csv(LINKS, sep='\t')
        self.genbed= pd.read_csv(self.genbed, sep='\t')[['#chrom', 'start', 'end', 'gene_name', 'gene_id', 'gene_biotype']]\
                        .rename(columns={'gene_name' : 'gene', 'gene_id' : 'gene_ID'}).copy()
        self.Head  = Head
        self.outpre= self.outdir + '/' + Head

        os.makedirs(self.outdt, exist_ok=True)
        return self

    def _getgene(self, _link, gene_bt=['protein_coding', 'lncRNA', 'miRNA'], minover=30):
        _chr, _start, _end = _link[0], min(_link[1:]), max(_link[1:])
        if gene_bt:
            set0 = ((self.genbed['#chrom'] == _chr) & (self.genbed['gene_biotype'].isin(gene_bt)))
        else:
            set0 = (self.genbed['#chrom'] == _chr) 

        set1 = ((self.genbed.start>= _start) & (self.genbed.end<= _end))
        set2 = ((self.genbed.start< _start)  & (self.genbed.end>= _start+minover))
        set3 = ((self.genbed.start<= _end-minover) & (self.genbed.end> _end))
        targetg = self.genbed[ (set0 & (set1 | set2 | set3)) ].copy()
        if _link[1:3] == sorted(_link[1:3]):
            targetg.sort_values(by=['#chrom', 'start', 'end'], ascending=[True, True,  True], inplace=True)
        else:
            targetg.sort_values(by=['#chrom', 'start', 'end'], ascending=[True, False,False], inplace=True)
        
        targetg.loc[targetg.start<_start, 'start'] = _start
        targetg.loc[targetg.end > _end,     'end'] =  _end
        return targetg

    def _getgbed(self, Links):
        _G = pd.concat([ self._getgene(i) for i in Links], axis=0)
        _G['#chrom'] = 'hs' + _G['#chrom']
        _G['length'] = _G['end'] - _G['start']
        _G['Color'] = [ 'color='+i for i in self.color_[:_G.shape[0]] ]
        _G.sort_values(by=['#chrom', 'length'], ascending=[True, False], inplace=True)
        _G[['#chrom', 'start', 'end', 'gene', 'Color']].to_csv(self.outdt + '/gene.text.txt', sep='\t', index=False, header=False)
        _G[['#chrom', 'start', 'end', 'Color']].to_csv(self.outdt + '/gene.tile.txt', sep='\t', index=False, header=False)

    def _getlink(self, _Link):
        Link, LinkA, LinkN = [], [], []
        LinkLen = 0
        for i in _Link.split(';'):
            _r = re.split('[:-]', i)
            _r = [str(_r[0])] + list(map(int, _r[1:]))
            _l = abs(_r[2]-_r[1])
            Link.append( _r )
            LinkA.append( [_r[0], min(_r[1:3]), max(_r[1:3])] )
            LinkN.append( [_r[0], LinkLen, LinkLen + _l] )
            LinkLen += _l
        return Link

    def _getlinks(self, _link):
        Drop  = self.LINKS[~(self.LINKS['#chrom'].isin(self.CHRK))]['LINKS'].tolist()
        self.LINKS = self.LINKS[~(self.LINKS.LINKS.isin(Drop))]
        Links = pd.concat([ self.LINKS[((self.LINKS['#chrom']==i[0]) & 
                                        (self.LINKS['start']==i[1]) &
                                        (self.LINKS['end']==i[2]) 
                                        )] #(self.LINKS['Supportsum']>=3)
                            for i in _link ], axis=0)
        print(Links)
        tileg, linkb = [], []
        K = sorted([self._getlink(n) for n in Links.BPs.unique()])
        for i, n in enumerate(K):
            for k,j in enumerate(n):
                tileg.append( ['hs'+j[0], j[1], j[2], 'color=%s;type=%s'%(self.Cole[i+k], i)])
            if len(n)==2:
                linkb.append([ 'hs'+n[0][0], n[0][1], n[0][1]+1, 
                               'hs'+n[1][0], n[1][2], n[1][2]-1, 'type=%s'%i])
        tileg = pd.DataFrame(tileg)
        linkb = pd.DataFrame(linkb)
        tileg.to_csv(self.outdt + '/bps.txt', sep='\t', index=False, header=False)
        linkb.to_csv(self.outdt + '/bps.link.txt', sep='\t', index=False, header=False)

    def CMD(self, INfile, OU, Head, Link, LINKS):
        self._getinfo( INfile, OU, Head, LINKS )
        self._getlinks(self._getlink(Link))
        self._getgbed(self._getlink(Link))
        #self._genebed()
        #self._getrna()
        #self._getkaryo1()

def Start91():
    INfile= '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/ecDNA.sample.info.20210225.txt'
    LINKS='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/COLON/EcDNA/20210303/04.EcRegion/All.circle.region.UpFilterTRF'
    Link='8:127738190-127824396'
    Cell='COLON'
    OU  ='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/20210303/SCircos/%s.%s/'%(Cell,Link)
    DT  = '20210303'
    Head='%s.%s.'%(Cell, Link)
    SCircos().CMD(INfile, OU, Head, Link, LINKS)

#Start91()

class ecDNAoverlap:
    def __init__(self):
        self.CHRS=[str(i) for i in range(1,23)] + ['MT','X','Y']
        self.CHRK=[str(i) for i in range(1,23)] + ['X','Y']
        self.PLMD=['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
    
    def _getpublic(self, cl='COLO320DM'):
        infile = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/WorkShell/Public.ecDNA.info.txt'
        indf = pd.read_csv(infile, sep='\t')
        indf = indf[((indf.Type=='Circular') & (indf.Sample==cl))]
        return indf

    def _getec(self, fl):
        FL = pd.read_csv(fl, sep='\t')
        D1 = FL[~(FL['#chrom'].isin(self.CHRS))]['LINKS'].tolist()
        FL = FL[~(FL.LINKS.isin(D1))]
        Fnum =len(FL.LINKS.unique())
        return FL
    
    def vennplot(self, vdict, out):
        from venn import venn
        fig, ax = plt.subplots(figsize=(8,8))
        ax = venn(vdict)
        plt.savefig( self.out, bbox_inches='tight')
        plt.close()

    def Intersect(self, da, db, f=0.1, F=0.1):
        da = da.copy()
        db = db.copy()
        da['#chrom'] = da['#chrom'].astype(str)
        db['#chrom'] = db['#chrom'].astype(str)
        Ca = list(da.columns)
        Cb = list(db.columns + '_b')

        da = bt.BedTool.from_dataframe(da)
        db = bt.BedTool.from_dataframe(db)

        bbed = da.intersect(db, s=False, S=False, wa=True, wb=True, f=f, F=F)\
                    .to_dataframe(disable_auto_names=True, header=None, names=Ca+Cb)

        bbed = bbed.infer_objects()
        #bbed['#chrom'] = bbed['#chrom'].astype(str)
        return bbed

    def IntersectA(self, da, db, dh,  f=0.1, F=0.1):
        da = da.copy()
        db = db.copy()
        da['#chrom'] = da['#chrom'].astype(str)
        db['#chrom'] = db['#chrom'].astype(str)
        dh['#chrom'] = dh['#chrom'].astype(str)
        Ca = list(da.columns)
        Cb = list(db.columns + '_b')
        Cc = list(dh.columns + '_c')

        da = bt.BedTool.from_dataframe(da)
        db = bt.BedTool.from_dataframe(db)
        dc = bt.BedTool.from_dataframe(dh)

        bbed = da.intersect((db,dc), s=False, S=False, wa=True, wb=True, f=f, F=F)\
                    .to_dataframe(disable_auto_names=True, header=None, names=Ca+Cb)

        bbed = bbed.infer_objects()
        #bbed['#chrom'] = bbed['#chrom'].astype(str)
        return bbed

    def Getcom(self, FL='COLON', CL='COLO320DM', Dat='20210405'):
        FLf  = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/%s.circos.xls'%(Dat, FL)
        pubs = self._getpublic(cl=CL)
        ecdf = self._getec(FLf)
        comm = self.Intersect(ecdf, pubs)
        cset = [ FL,len(ecdf.LINKS.unique()), len(comm.LINKS.unique()), 
                    len(pubs.coord.unique()), len(comm['coord_b'].unique()) ]
        return cset

    def GetU2OS(self, Dat='20210405'):
        Pac = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/U2OS.CCS.circos.xls'%(Dat)
        Nao = '/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/U2OS.circos.xls'%(Dat)
        Pac = self._getec(Pac)
        Nao = self._getec(Nao)
        comm = self.Intersect(Nao, Pac)
        cset = ['O2OS', len(Nao.LINKS.unique()), len(comm.LINKS.unique()), 
                        len(Pac.LINKS.unique()), len(comm['LINKS_b'].unique()) ]
        return cset

    def Get3CLs(self, Dat='20210405'):
        U2OS ='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/U2OS.circos.xls'%(Dat)
        PC3  ='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/PC3.circos.xls'%(Dat)
        COLON='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/Analyze/%s/EcDNA/COLON.circos.xls'%(Dat)

        U2OS = self._getec(U2OS)
        PC3  = self._getec(PC3)
        COLON= self._getec(COLON)
        
        U2P  = self.Intersect(U2OS, PC3)
        U2C  = self.Intersect(U2OS, COLON)
        P2C  = self.Intersect(PC3, COLON)
        P2U  = self.Intersect(PC3, U2OS)
        C2P  = self.Intersect(COLON, PC3)
        C2U  = self.Intersect(COLON, U2OS)

        U2P2C = self.Intersect(U2P.iloc[:, :15], COLON)
        P2U2C = self.Intersect(P2U.iloc[:, :15], COLON)

        C2P2U = self.Intersect(C2P.iloc[:, :15], U2OS)
        P2C2U = self.Intersect(P2C.iloc[:, :15], U2OS)

        U2C2P = self.Intersect(U2C.iloc[:, :15], PC3)
        C2U2P = self.Intersect(C2U.iloc[:, :15], PC3)

        cset = [len(U2OS.LINKS.unique()),
                len(PC3.LINKS.unique()),
                len(COLON.LINKS.unique()),

                len(U2P.LINKS.unique()),
                len(U2P['LINKS_b'].unique()),
                len(U2C.LINKS.unique()),
                len(U2C['LINKS_b'].unique()),
                len(P2C.LINKS.unique()),
                len(P2C['LINKS_b'].unique()),

                len(U2P2C.LINKS.unique()),
                len(U2P2C['LINKS_b'].unique()),
                len(P2U2C.LINKS.unique()),
                len(P2U2C['LINKS_b'].unique()),

                len(C2P2U.LINKS.unique()),
                len(C2P2U['LINKS_b'].unique()),
                len(P2C2U.LINKS.unique()),
                len(P2C2U['LINKS_b'].unique()),

                len(U2C2P.LINKS.unique()),
                len(U2C2P['LINKS_b'].unique()),
                len(C2U2P.LINKS.unique()),
                len(C2U2P['LINKS_b'].unique()),
                ]
        cnam = ['U2OS', 'PC3', 'COLON', 
                'U2P_U', 'U2P_P', 'U2C_U', 'U2C_C', 'P2C_P', 'P2C_C', 
                'U2P2C_U2P', 'U2P2C_C', 'P2U2C_P2U', 'P2U2C_C', 
                'C2P2U_C2P', 'C2P2U_U', 'P2C2U_P2C', 'P2C2U_U', 
                'U2C2P_U2C',  'U2C2P_P', 'C2U2P_C2U', 'C2U2P_P']
        comp = pd.DataFrame(cset, index=cnam)
        print(comp)

    def GetPub(self):
        K = [ self.Getcom(FL='COLON', CL='COLO320DM'), self.Getcom(FL='PC3', CL='PC3'), self.GetU2OS()]
        print(K)
        #self._getlinknum(FL)
        self.Get3CLs()
    
def Start101():
    ecDNAoverlap().GetPub()

#Start101()
