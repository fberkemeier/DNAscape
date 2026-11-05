/* Conditional fork transport and hazard. No simulation or Python callback per cell. */
#include <math.h>
#include <stddef.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static inline double larger(double a, double b) { return a>b ? a : b; }
static inline double smaller(double a, double b) { return a<b ? a : b; }

static double slope(double a, double b) {
    if (a*b <= 0.) return 0.;
    return a>0. ? smaller(a,b) : -smaller(-a,-b);
}

static void rhs(size_t n, int periodic, int counts, double dx, double w, double floor,
                const double *i0, const double *i1, const double *v0, const double *v1,
                int is, int vs, const double *y, double *out, double *flux) {
    for (size_t j=0; j<n; ++j) {
        double v=(1.-w)*v0[j*vs]+w*v1[j*vs];
        double rate=(1.-w)*i0[j*is]+w*i1[j*is];
        double r=larger(y[n+j],0.), l=larger(y[2*n+j],0.);
        out[j]=v*(r+l);
        if (counts) {
            double s=exp(-larger(y[j],0.));
            out[3*n+j]=rate*s;
            out[4*n+j]=s>floor ? 2.*v*r*l*s : 0.;
        }
    }
    for (int direction=1; direction>=-1; direction-=2) {
        const double *g=y+(direction==1 ? n : 2*n);
        for (size_t j=0; j<n; ++j) {
            size_t p=j ? j-1 : n-1, q=j+1<n ? j+1 : 0;
            double prev=g[p], next=g[q];
            if (!periodic && j==0) prev=direction==1 ? -g[j] : 2.*g[j]-g[n>1 ? 1 : 0];
            if (!periodic && j==n-1) next=direction==-1 ? -g[j] : 2.*g[j]-g[n>1 ? n-2 : 0];
            double m=slope(g[j]-prev,next-g[j]);
            double v=(1.-w)*v0[j*vs]+w*v1[j*vs];
            size_t neighbor=direction==1 ? q : p;
            double vn=(1.-w)*v0[neighbor*vs]+w*v1[neighbor*vs];
            double vf=.5*(v+vn);
            if (!periodic && ((direction==1 && j==n-1) || (direction==-1 && j==0))) vf=v;
            flux[j]=vf*larger(g[j]+direction*.5*m,0.);
        }
        double *dg=out+(direction==1 ? n : 2*n);
        for (size_t j=0; j<n; ++j) {
            size_t neighbor=direction==1 ? (j ? j-1 : n-1) : (j+1<n ? j+1 : 0);
            double incoming=flux[neighbor];
            if (!periodic && ((direction==1 && j==0) || (direction==-1 && j==n-1))) incoming=0.;
            double rate=(1.-w)*i0[j*is]+w*i1[j*is];
            dg[j]=rate+(incoming-flux[j])/dx;
        }
    }
}

EXPORT void event_interval(size_t n, size_t steps, size_t first, size_t count,
                           double interval, double dx, int periodic, int counts, double floor,
                           int is, int vs, const double *i0, const double *i1,
                           const double *v0, const double *v1, double *state, double *work) {
    size_t m=(counts ? 5 : 3)*n;
    double h=interval/steps;
    double *k1=work, *k2=k1+m, *k3=k2+m, *k4=k3+m, *tmp=k4+m, *flux=tmp+m;
    for (size_t step=first; step<first+count; ++step) {
        double w=(double)step/steps, half=w+.5/steps, end=w+1./steps;
        rhs(n,periodic,counts,dx,w,floor,i0,i1,v0,v1,is,vs,state,k1,flux);
        if (!counts) {
            /* SSP RK2 is sufficient for sampled fields, with the same
               second-order conservative flux and positive CFL bound. */
            for (size_t j=0;j<m;++j) tmp[j]=state[j]+h*k1[j];
            rhs(n,periodic,counts,dx,end,floor,i0,i1,v0,v1,is,vs,tmp,k2,flux);
            for (size_t j=0;j<m;++j) state[j]=larger(0.,state[j]+.5*h*(k1[j]+k2[j]));
            continue;
        }
        for (size_t j=0;j<m;++j) tmp[j]=state[j]+.5*h*k1[j];
        rhs(n,periodic,counts,dx,half,floor,i0,i1,v0,v1,is,vs,tmp,k2,flux);
        for (size_t j=0;j<m;++j) tmp[j]=state[j]+.5*h*k2[j];
        rhs(n,periodic,counts,dx,half,floor,i0,i1,v0,v1,is,vs,tmp,k3,flux);
        for (size_t j=0;j<m;++j) tmp[j]=state[j]+h*k3[j];
        rhs(n,periodic,counts,dx,end,floor,i0,i1,v0,v1,is,vs,tmp,k4,flux);
        for (size_t j=0;j<m;++j) state[j]=larger(0.,state[j]+h/6.*(k1[j]+2.*k2[j]+2.*k3[j]+k4[j]));
    }
}
