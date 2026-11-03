/* Cell-exact stationary cone points and their discrete transpose.
 * No fast-math, model approximations, dense Jacobians or distributed binaries.
 */
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

typedef struct {size_t cell; double h, turns, mass, qpart, M, Q;} Point;
typedef struct {size_t n; double dx, cap, length; int periodic;
 const double *rate,*slow,*y,*mass,*moment,*yc,*mc,*qc;size_t hints[2];} Model;

static double inverse(Model *a,double query,int side) {
 double turns=a->periodic?floor(query/a->y[a->n]):0.;
 double local=query-turns*a->y[a->n];
 if(!a->periodic) local=fmax(0.,fmin(a->y[a->n],local));
 size_t hint=a->hints[side],lo,hi,step=1;
 if(local>=a->y[hint]){
  lo=hint;hi=hint<a->n?hint+1:a->n;
  while(hi<a->n && a->y[hi]<=local){lo=hi;step*=2;hi=step>a->n-hi?a->n:hi+step;}
 }else{
  hi=hint;lo=hint?hint-1:0;
  while(lo && a->y[lo]>local){hi=lo;step*=2;lo=lo>step?lo-step:0;}
 }
 while(lo<hi){size_t mid=lo+(hi-lo)/2;if(a->y[mid]<=local)lo=mid+1;else hi=mid;}
 size_t cell=lo?lo-1:0;if(cell>=a->n)cell=a->n-1;
 a->hints[side]=cell;
 return turns*a->length+cell*a->dx+(local-a->y[cell])/a->slow[cell];
}
static Point prefix(const Model *a,double position) {
 Point p; p.turns=a->periodic?floor(position/a->length):0.;
 double local=position-p.turns*a->length;
 if(!a->periodic)local=fmax(0.,fmin(a->length,local));
 double index=floor(local/a->dx);
 p.cell=index<=0.?0:((size_t)index>=a->n?a->n-1:(size_t)index);
 p.h=local-p.cell*a->dx;
 p.qpart=a->y[p.cell]*p.h+.5*a->slow[p.cell]*p.h*p.h;
 p.mass=a->mass[p.cell]+p.h*a->rate[p.cell];
 p.M=p.mass+p.turns*a->mass[a->n];
 p.Q=a->moment[p.cell]+p.qpart*a->rate[p.cell]+p.turns*a->moment[a->n]
     +.5*p.turns*(p.turns-1.)*a->y[a->n]*a->mass[a->n]+p.turns*a->y[a->n]*p.mass;
 return p;
}
static void ends(Model *a,double t,size_t receiver,Point *l,Point *r) {
 double x=(receiver+.5)*a->dx;
 *l=prefix(a,fmax(x-a->cap,inverse(a,a->yc[receiver]-t,0)));
 *r=prefix(a,fmin(x+a->cap,inverse(a,a->yc[receiver]+t,1)));
}
static double hazard(const Model *a,double t,size_t receiver,Point l,Point r) {
 return t*(r.M-l.M)-(a->yc[receiver]*(2*a->mc[receiver]-l.M-r.M)+l.Q+r.Q-2*a->qc[receiver]);
}
EXPORT unsigned int stationary_timing_abi(void){return 1;}
EXPORT void stationary_timing_points(size_t n,size_t m,double dx,double cap,int periodic,
 const double *rate,const double *slow,const double *y,const double *mass,const double *moment,
 const double *yc,const double *mc,const double *qc,const double *times,const int64_t *receivers,double *out) {
 Model a={n,dx,cap,n*dx,periodic,rate,slow,y,mass,moment,yc,mc,qc};
 for(size_t j=0;j<m;j++){Point l,r;size_t index=(size_t)receivers[j];ends(&a,times[j],index,&l,&r);
  out[j]=fmax(0.,hazard(&a,times[j],index,l,r));}
}
static void accumulate(const Model *a,Point p,double mg,double qg,int speed,
 double *gm,double *gq,double *gy,double *gi,double *gw) {
 double lm=mg+p.turns*a->y[a->n]*qg;
 gm[p.cell]+=lm;gq[p.cell]+=qg;
 gm[a->n]+=p.turns*mg+.5*p.turns*(p.turns-1.)*a->y[a->n]*qg;
 gq[a->n]+=p.turns*qg;
 gi[p.cell]+=p.h*lm+p.qpart*qg;
 if(speed){gy[p.cell]+=a->rate[p.cell]*p.h*qg;gw[p.cell]+=.5*a->rate[p.cell]*p.h*p.h*qg;
  gy[a->n]+=(.5*p.turns*(p.turns-1.)*a->mass[a->n]+p.turns*p.mass)*qg;}
}
EXPORT void stationary_timing_pullback(size_t n,size_t m,double dx,double cap,int periodic,int speed,
 const double *rate,const double *slow,const double *y,const double *mass,const double *moment,
 const double *yc,const double *mc,const double *qc,const double *times,const int64_t *receivers,
 const double *g,const double *gmass,double *gm,double *gq,double *gy,double *gi,double *gw,double *greceiver) {
 Model a={n,dx,cap,n*dx,periodic,rate,slow,y,mass,moment,yc,mc,qc};
 for(size_t j=0;j<m;j++){
  size_t index=(size_t)receivers[j];Point l,r;ends(&a,times[j],index,&l,&r);
  double value=hazard(&a,times[j],index,l,r)>0.?g[j]:0.;double center=yc[index];
  accumulate(&a,l,value*(-times[j]+center)-gmass[j],-value,speed,gm,gq,gy,gi,gw);
  accumulate(&a,r,value*(times[j]+center)+gmass[j],-value,speed,gm,gq,gy,gi,gw);
  Point c=prefix(&a,(index+.5)*dx);
  accumulate(&a,c,-2*value*center,2*value,speed,gm,gq,gy,gi,gw);
  if(speed)greceiver[index]+=value*(l.M+r.M-2*mc[index]);
 }
}
