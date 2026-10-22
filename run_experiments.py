"""Run all experiments; no network, proprietary data, or manual figure edits."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv, json, platform, hashlib
import numpy as np
import scipy, sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from bsi import labeled_bsi, distributions_bsi
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'fig'; OUT=ROOT/'results'
FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight','axes.labelsize':9})
BLUE='#2463A6'; ORANGE='#D27B25'; GREEN='#34876D'; GRAY='#777777'

def savecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def savefig(fig,name):
    fig.savefig(FIG/(name+'.pdf')); fig.savefig(FIG/(name+'.png'),dpi=220); plt.close(fig)
def summarize(a):
    return np.mean(a,axis=0),np.quantile(a,.025,axis=0),np.quantile(a,.975,axis=0)
def mixture(rng,N,gamma):
    p=np.array([.7,.2,.1]); counts=np.rint(N*p).astype(int)
    means=np.array([[-6,0],[0,0],[6,0]])
    X=np.vstack([rng.normal(size=(n,2))*p[k]**(gamma/2)+means[k] for k,n in enumerate(counts)])
    y=np.repeat(np.arange(3),counts)
    return X,y

# Study 1: known population target, sampling variability, legacy normalization.
p=np.array([.7,.2,.1]); gammas=np.linspace(-1,2,61); R=100; N=3000
samples=[]; rows=[]
for g in gammas:
    vals=[]; old=[]
    for rep in range(R):
        X,y=mixture(np.random.default_rng(10000+rep),N,g)
        result=labeled_bsi(X,y); vals.append(result['bsi'])
        raw=np.exp(result['log_extents'])*(result['counts']-1) # d=2
        old.append(distributions_bsi(result['p'],raw/raw.sum()))
        rows.append(dict(gamma=g,replicate=rep,N=N,bsi=vals[-1],raw_svd_bsi=old[-1]))
    q=p**g; q/=q.sum(); samples.append((g,distributions_bsi(p,q),*summarize(vals),np.mean(old)))
samples=np.array(samples); savecsv('synthetic_replicates.csv',rows)
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
ax[0].plot(samples[:,0],samples[:,1],color=BLUE,label='Population target')
ax[0].fill_between(samples[:,0],samples[:,3],samples[:,4],color=BLUE,alpha=.18,label='95% replicate range')
ax[0].plot(samples[:,0],samples[:,2],'--',color=GREEN,label='Covariance estimate')
ax[0].set(xlabel=r'Spread exponent $\gamma$',ylabel='BSI',title='(a) Controlled density imbalance',ylim=(.2,1.025)); ax[0].legend(fontsize=7,loc='lower right')
ax[1].plot(samples[:,0],samples[:,2],color=BLUE,label='Covariance extent')
ax[1].plot(samples[:,0],samples[:,5],color=ORANGE,label='Raw SVD product')
ax[1].axvline(1,color=GRAY,lw=.8,ls=':');ax[1].axvline(0,color=GRAY,lw=.8,ls=':')
ax[1].set(xlabel=r'Spread exponent $\gamma$',ylabel='BSI',title='(b) Sample-size normalization',ylim=(.2,1.025));ax[1].legend(fontsize=7,loc='lower right')
sep=[]
rng=np.random.default_rng(200); X0=rng.normal(size=(600,2)); y=np.repeat([0,1,2],200)
for a in np.linspace(0,8,33):
    X=X0+np.array([[-a,0],[0,0],[a,0]])[y]
    sep.append(dict(separation=a,bsi=labeled_bsi(X,y)['bsi'],silhouette=silhouette_score(X,y)))
ax[2].plot([r['separation'] for r in sep],[r['bsi'] for r in sep],color=BLUE,label='BSI')
ax[2].plot([r['separation'] for r in sep],[r['silhouette'] for r in sep],color=ORANGE,label='Silhouette')
ax[2].set(xlabel='Distance between adjacent means',ylabel='Index value',title='(c) Separation is a different property',ylim=(-.1,1.05));ax[2].legend(fontsize=7)
fig.tight_layout(w_pad=2);savefig(fig,'study1_controlled');savecsv('separation.csv',sep)
convergence=[]
for N in [150,600,2400,9600]:
    for rep in range(200):
        X,y=mixture(np.random.default_rng(30000+rep),N,0)
        value=labeled_bsi(X,y)['bsi']; target=distributions_bsi(p,np.ones(3)/3)
        convergence.append(dict(N=N,replicate=rep,bsi=value,target=target,error=value-target))
savecsv('sample_size.csv',convergence)
# Study 2: exact region volumes and nonnested covariance partitions.
ks=2**np.arange(7); refinement=[]
for k in ks:
    e=np.linspace(0,1,k+1); mass=np.diff(e**2); q=np.diff(e)
    refinement.append(dict(K=int(k),bsi=distributions_bsi(mass,q)))
ts=np.linspace(.01,.99,199)
boundary=[dict(t=t,bsi=distributions_bsi([t*t,1-t*t],[t,1-t])) for t in ts]
X,y=mixture(np.random.default_rng(410),2400,0); partition=[]
for k in range(2,9):
    for seed in range(30):
        fit=KMeans(n_clusters=k,n_init=1,random_state=seed,algorithm='lloyd').fit(X)
        partition.append(dict(K=k,seed=seed,bsi=labeled_bsi(X,fit.labels_)['bsi'],inertia=fit.inertia_))
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
ax[0].plot(ks,[r['bsi'] for r in refinement],'o-',color=BLUE)
ax[0].set_xscale('log',base=2); ax[0].set_xticks(ks,labels=ks)
ax[0].set(xlabel='Number of nested intervals K',ylabel='BSI',title='(a) Additive region volumes')
ax[1].plot(ts,[r['bsi'] for r in boundary],color=BLUE)
ax[1].set(xlabel='Two-state boundary t',ylabel='BSI',title='(b) Fixed K, changing boundary')
for k in range(2,9):
    v=np.array([r['bsi'] for r in partition if r['K']==k]); ax[2].scatter(np.full(len(v),k),v,color=BLUE,alpha=.22,s=9)
ax[2].plot(range(2,9),[np.median([r['bsi'] for r in partition if r['K']==k]) for k in range(2,9)],'o-',color=ORANGE,label='Median over 30 initializations')
ax[2].set(xlabel='K-means cluster count K',ylabel='BSI',title='(c) Recomputed covariance extents');ax[2].legend(fontsize=7)
fig.tight_layout(w_pad=2);savefig(fig,'study2_partition')
savecsv('region_refinement.csv',refinement);savecsv('region_boundary.csv',boundary);savecsv('kmeans_partition.csv',partition)
# Study 3: Iris, initialization, estimator dependence; no selection by BSI.
iris=load_iris(); X,y=iris.data,iris.target
np.savetxt(OUT/'iris_data.csv',np.column_stack([X,y]),delimiter=',',header='sepal_length,sepal_width,petal_length,petal_width,species',comments='')
fits=[]; initial=[]
for seed in range(50):
    fit=KMeans(n_clusters=3,n_init=1,random_state=seed,algorithm='lloyd').fit(X); fits.append(fit)
    initial.append(dict(seed=seed,inertia=fit.inertia_,bsi=labeled_bsi(X,fit.labels_)['bsi'],ari=adjusted_rand_score(y,fit.labels_)))
best=fits[int(np.argmin([f.inertia_ for f in fits]))]
np.savetxt(OUT/'iris_selected_labels.csv',best.labels_,fmt='%d',header='kmeans_label',comments='')
records=[]; components=[]; boot=[]
for name,labels in [('Species',y),('K-means',best.labels_)]:
    for est in ['covariance','hull']:
        result=labeled_bsi(X,labels,est)
        records.append(dict(labeling=name,extent=est,bsi=result['bsi'],silhouette=silhouette_score(X,labels),calinski_harabasz=calinski_harabasz_score(X,labels),davies_bouldin=davies_bouldin_score(X,labels),ari=adjusted_rand_score(y,labels)))
        for k in range(3): components.append(dict(labeling=name,extent=est,state=int(k),count=int(result['counts'][k]),p=result['p'][k],q=result['q'][k]))
    rng=np.random.default_rng(550 if name=='Species' else 551)
    for rep in range(1000):
        indices=np.concatenate([rng.choice(np.flatnonzero(labels==k),sum(labels==k),replace=True) for k in range(3)])
        boot.append(dict(labeling=name,replicate=rep,bsi=labeled_bsi(X[indices],labels[indices])['bsi']))
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
colors=[BLUE,ORANGE,GREEN]
for k in range(3):
    ax[0].scatter(X[y==k,2],X[y==k,3],s=15,color=colors[k],label=iris.target_names[k],alpha=.8,edgecolors='none')
ax[0].set(xlabel='Petal length (cm)',ylabel='Petal width (cm)',title='(a) Iris species (2 of 4 features)');ax[0].legend(fontsize=7)
r=labeled_bsi(X,y);pos=np.arange(3)
ax[1].bar(pos-.17,r['p'],width=.34,color=GRAY,label='Mass p')
ax[1].bar(pos+.17,r['q'],width=.34,color=BLUE,label='Covariance extent q')
ax[1].set_xticks(pos,['setosa','versicolor','virginica'],rotation=18)
ax[1].set(ylabel='Normalized share',title='(b) Equal counts, unequal extents');ax[1].legend(fontsize=7)
for name,col in [('Species',BLUE),('K-means',ORANGE)]:
    vals=[r['bsi'] for r in boot if r['labeling']==name]
    ax[2].hist(vals,bins=35,histtype='step',lw=1.5,color=col,label=name,density=True)
    value=[r['bsi'] for r in records if r['labeling']==name and r['extent']=='covariance'][0]
    ax[2].axvline(value,color=col,ls='--',lw=1)
ax[2].set(xlabel='BSI',ylabel='Bootstrap density',title='(c) Fixed-label stratified bootstrap');ax[2].legend(fontsize=7)
fig.tight_layout(w_pad=2);savefig(fig,'study3_iris')
savecsv('iris_metrics.csv',records);savecsv('iris_components.csv',components);savecsv('iris_initializations.csv',initial);savecsv('iris_bootstrap.csv',boot)
summary={'synthetic_gamma0':samples[20].tolist(),'synthetic_gamma1':samples[40].tolist(),'refinement':refinement,'iris':records,'iris_components':components,'iris_best_seed':int(np.argmin([f.inertia_ for f in fits])),'iris_inertia':best.inertia_,'iris_bootstrap':{name:dict(zip(['mean','lower','upper'],map(float,summarize([r['bsi'] for r in boot if r['labeling']==name])))) for name in ['Species','K-means']},'convergence':{str(N):float(np.sqrt(np.mean([r['error']**2 for r in convergence if r['N']==N]))) for N in [150,600,2400,9600]}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'iris_sha256':hashlib.sha256(X.tobytes()).hexdigest()},indent=2))
print(json.dumps(summary,indent=2))
