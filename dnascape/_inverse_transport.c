/* Exact finite-volume forward/tangent transport used by inverse.py.
 * No fast-math: clipping and branch derivatives must match the NumPy path.
 * One call advances a single time interval so Python retains cancellation.
 */
#include <stddef.h>
#include <string.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static double larger(double a, double b) { return a > b ? a : b; }
static double smaller(double a, double b) { return a < b ? a : b; }

EXPORT unsigned int inverse_transport_abi_version(void) { return 2; }

EXPORT void inverse_transport_interval_v2(
    size_t n, size_t p, size_t substeps, size_t first_step, size_t step_count,
    double interval, double resolution,
    int periodic, size_t is0, size_t is1, size_t vs0, size_t vs1,
    const double *i0, const double *i1, const double *v0, const double *v1,
    const double *di0, const double *di1, const double *dv0, const double *dv1,
    double *restrict f, double *restrict right, double *restrict left, double *restrict timing,
    double *restrict df, double *restrict dright, double *restrict dleft, double *restrict dtiming, double *restrict scratch)
{
    const size_t np = n * p;
    double *rf = scratch;
    double *lf = rf + n;
    double *old_f = lf + n;
    double *drf = old_f + n;
    double *dlf = drf + np;
    double *old_df = dlf + np;
    const double dt = interval / (double)substeps;
    const double dt_dx = dt / resolution;
    const double inv_dt = 1.0 / dt;
    if (first_step == 0) {
        memcpy(old_f, f, n * sizeof(double));
        if (p) memcpy(old_df, df, np * sizeof(double));
    }
    for (size_t step = first_step; step < first_step + step_count; ++step) {
        const double weight = ((double)step + 0.5) / (double)substeps;
        const double complement = 1.0 - weight;
        /* Keep all old neighbour fluxes before updating any positions. */
        for (size_t x = 0; x < n; ++x) {
            if (f[x] >= 1.0 - 1e-12) {
                rf[x] = lf[x] = 0.0;
                if (p) {
                    memset(drf + x * p, 0, p * sizeof(double));
                    memset(dlf + x * p, 0, p * sizeof(double));
                }
                continue;
            }
            const double velocity = complement * v0[x * vs0] + weight * v1[x * vs1];
            rf[x] = velocity * right[x];
            lf[x] = -velocity * left[x];
            for (size_t j = 0; j < p; ++j) {
                const size_t k = x * p + j;
                const double dv = complement * dv0[k] + weight * dv1[k];
                drf[k] = dv * right[x] + velocity * dright[k];
                dlf[k] = -dv * left[x] - velocity * dleft[k];
            }
        }
        for (size_t x = 0; x < n; ++x) {
            const size_t previous = x ? x - 1 : n - 1;
            const size_t next = x + 1 < n ? x + 1 : 0;
            const int incoming = periodic || x > 0;
            const int outgoing = periodic || x + 1 < n;
            const double initiation = complement * i0[x * is0] + weight * i1[x * is1];
            const double velocity = complement * v0[x * vs0] + weight * v1[x * vs1];
            const double rin = incoming ? rf[previous] : 0.0;
            const double lout = outgoing ? lf[next] : 0.0;
            const double rt = larger(right[x] - dt * (rf[x] - rin) / resolution, 0.0);
            const double lt = larger(left[x] - dt * (lout - lf[x]) / resolution, 0.0);
            const double u = larger(1.0 - f[x], 0.0);
            const double source = initiation * u;
            const double denominator = larger(u, 1e-12);
            const double raw_sink = 2.0 * velocity * rt * lt / denominator;
            const double max_sink = source + smaller(rt, lt) / dt;
            const double sink = smaller(raw_sink, max_sink);
            const double rn = larger(rt + dt * (source - sink), 0.0);
            const double ln = larger(lt + dt * (source - sink), 0.0);
            const double total = 0.5 * (rt + rn + lt + ln);
            const double rate = velocity * total;
            const double new_f = smaller(larger(f[x] + smaller(dt * rate, u), 0.0), 1.0);
            const int completed = new_f >= 1.0 - 1e-12;
            const double two_inv_denominator = 2.0 / denominator;
            const double sink_inv_denominator = raw_sink / denominator;
            if (completed && p) {
                memset(df + x * p, 0, p * sizeof(double));
                memset(dright + x * p, 0, p * sizeof(double));
                memset(dleft + x * p, 0, p * sizeof(double));
            }
            for (size_t j = 0; !completed && j < p; ++j) {
                const size_t k = x * p + j;
                const double di = complement * di0[k] + weight * di1[k];
                const double dv = complement * dv0[k] + weight * dv1[k];
                const double drin = incoming ? drf[previous * p + j] : 0.0;
                const double dlout = outgoing ? dlf[next * p + j] : 0.0;
                const double drt = (dright[k] - dt_dx * (drf[k] - drin)) * (rt > 0.0);
                const double dlt = (dleft[k] - dt_dx * (dlout - dlf[k])) * (lt > 0.0);
                const double du = -df[k] * (u > 0.0);
                const double dsource = di * u + initiation * du;
                double dsink = two_inv_denominator * (dv * (rt * lt) + velocity * (drt * lt + rt * dlt))
                    - sink_inv_denominator * du * (u > 1e-12);
                const double dmax_sink = dsource + (rt <= lt ? drt : dlt) * inv_dt;
                if (!(raw_sink <= max_sink)) dsink = dmax_sink;
                const double drn = (drt + dt * (dsource - dsink)) * (rn > 0.0);
                const double dln = (dlt + dt * (dsource - dsink)) * (ln > 0.0);
                const double drate = dv * total + 0.5 * velocity * (drt + drn + dlt + dln);
                df[k] += dt * rate < u ? dt * drate : du;
                dright[k] = completed ? 0.0 : drn;
                dleft[k] = completed ? 0.0 : dln;
                if (completed) df[k] = 0.0;
            }
            f[x] = new_f;
            right[x] = completed ? 0.0 : rn;
            left[x] = completed ? 0.0 : ln;
        }
    }
    if (first_step + step_count == substeps) {
        for (size_t x = 0; x < n; ++x)
            timing[x] += interval * (1.0 - 0.5 * (old_f[x] + f[x]));
        for (size_t k = 0; k < np; ++k)
            dtiming[k] -= interval * 0.5 * (old_df[k] + df[k]);
    }
}
