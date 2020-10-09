/*
 *
 * [xnew invK] = recommendSample(x,y,N,invK,lb,ub,nDim,ktype,ksigma,eps,msrSigma,acqFuncType)
 */
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "mex.h"
#include "nlopt.h"

#define MAX_DATA 1000
#define min(x1,x2) (((x1)<(x2))? (x1): (x2))
#define max(x1,x2) (((x1)>(x2))? (x1): (x2))
#define MATH_PI 3.14159
#define TINY 1.0e-20

double distv(double *x1, double *x2);
void print_vector(int n, double* v);
void print_matrix(int m, int n, double** v);


int nDim;
double OptTime;
double gValmax = 0.0;

typedef struct{
    int ktype;
    double nu;
    double scale;
    double kvar;
    double rqalpha;
} kernelOptions;

void kernelFunction(double *kvec, double *xnew, double **x, int N, kernelOptions koptions);

typedef struct{
    int criteria;
    double msrSigma2;
    double *lb;
    double *ub;
    double eps;
    int if_global;
}bayesOptions;

typedef struct{
    double *x;
    double *y;
    double *invK;
    double *kvec;
    int N;
    double ymax;
    int onlykernel;
    kernelOptions koptions;
    bayesOptions boptions;
}acqFuncData;


double acqFunc(unsigned n, const double *x, double *grad, void *my_func_data);
void recommendSample(double *maxf, double *xnew, double *x, double *y, double ymax, int N, double *kvec, double *invK, kernelOptions koptions, bayesOptions boptions);


/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    
    /*checking number of arguments*/
    
    if (nrhs != 19) {
        mexErrMsgIdAndTxt("MyToolbox:recommendSample:nrhs", "19 inputs required.");
    }
    
    /* variable declarations here */
    
    double *x = mxGetPr(prhs[0]);
    double *y = mxGetPr(prhs[1]);
    double ymax = mxGetScalar(prhs[2]);
    int N = mxGetScalar(prhs[3]);
    double *kvec = mxGetPr(prhs[4]);
    double *invK = mxGetPr(prhs[5]);
    double *lb = mxGetPr(prhs[6]);
    double *ub = mxGetPr(prhs[7]);
    nDim = mxGetScalar(prhs[8]);
    OptTime = mxGetScalar(prhs[9]);
    int ktype = mxGetScalar(prhs[10]);
    double ksigma = mxGetScalar(prhs[11]);
    double kvar = mxGetScalar(prhs[12]);
    double rqalpha = mxGetScalar(prhs[13]);
    double eps = mxGetScalar(prhs[14]);
    double msrSigma2 = mxGetScalar(prhs[15]);
    int acqFuncType = mxGetScalar(prhs[16]);
    double *xinit = mxGetPr(prhs[17]);
    int if_global = mxGetScalar(prhs[18]);
    
    /* options here*/
    kernelOptions koptions;
    koptions.ktype = ktype;
    koptions.nu = 0;
    koptions.scale = ksigma;
    koptions.kvar = kvar;
    koptions.rqalpha = rqalpha;
    
    bayesOptions boptions;
    boptions.criteria = acqFuncType;
    boptions.msrSigma2 = msrSigma2;
    boptions.lb = lb;
    boptions.ub = ub;
    boptions.eps = eps;
    boptions.if_global = if_global;
    
    /* code here */
    
    nlhs = 2;
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *maxf = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, nDim, mxREAL);
    double *xnew = mxGetPr(plhs[1]);
    for (int ii = 0; ii < nDim; ii++)
        xnew[ii] = xinit[ii];
    recommendSample(maxf, xnew, x, y, ymax, N, kvec, invK, koptions, boptions);
    mxSetPr(plhs[0], maxf);
    mxSetPr(plhs[1], xnew);
    
    return;
}


double acqFunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
    acqFuncData *param = static_cast<acqFuncData*>(my_func_data);
    int N = param->N;
    
    double *xmat = param->x;
    
    if (param->koptions.ktype == 1) //SE kernel
    {
        for (int ii = 0; ii < N; ii++)
        {
            double dist = 0;
            for (int jj = 0; jj < nDim; jj++)
                dist = dist + (x[jj] - xmat[ii + jj*N]) * (x[jj] - xmat[ii + jj*N]);
            param->kvec[ii] = exp(-0.5*dist / (param->koptions.scale*param->koptions.scale));
        }
    }
    else
    {
        printf("Undefined kernel type\n");
        exit(0);
    }
    
    
    
    double gVal;
    double sm1 = 0;
    double mux = 0;
    for (int ii = 0; ii < N; ii++)
    {
        for (int jj = 0; jj < N; jj++)
        {
            sm1 = sm1 + param->kvec[ii] * param->invK[ii + jj*N] * param->kvec[jj];
            mux = mux + param->kvec[ii] * param->invK[ii + jj*N] * param->y[jj];
        }
    }
    
    double kTinvKk = sm1;
    double sigmax = sqrt((1 + param->boptions.msrSigma2) - min(sm1, 1));
    double ythr = param->ymax;
    
    //used by EI
    double z = (mux - ythr) / sigmax;
    double Phiz = 0.5 + 0.5 * erf(z / 1.4142);
    double phiz = (exp(-1 * z*z / 2) / 2.506628);
    
    
    double nu = 1.0;// used by UCB
    
    //Hardcoded for EI and no grad
// 	if (sigmax > 0)
// 	{
// 		gVal = (mux - ythr)*Phiz + sigmax*phiz;
// 	}
// 	else
// 	{
// 		gVal = 0;
// 	}
    
    switch (param->boptions.criteria)
    {
        case 1:
            //EI
            if (sigmax > 0)
            {
                gVal = (mux - ythr)*Phiz + sigmax*phiz;
            }
            else
            {
                gVal = 0;
            }
            break;
        case 2:
            //PI
            if (sigmax > 0)
            {
                gVal = 1 - 0.5 * (1 + erf((mux - ythr) / (sigmax * 1.4142)));
            }
            else
            {
                gVal = 0;
            }
            break;
        case 3:
            //UCB
            
            if (sigmax > 0)
            {
                gVal = mux + (sqrt(nu*param->boptions.eps)*sigmax);
            }
            else
            {
                gVal = 0;
            }
            break;
        default:
            //EI
            if (sigmax > 0)
            {
                gVal = (mux - ythr)*Phiz + sigmax*phiz;
            }
            else
            {
                gVal = 0;
            }
            break;
    }
    if (grad)
    {
        if (param->boptions.criteria==1)
        {
                if (sigmax > 0)
                {
                    for (int dd = 0; dd < nDim; dd++)
                    {
                        double muxdx = 0.0, sigmaxdx = 0.0;
                        for (int ii = 0; ii < N; ii++)
                        {
                            double temp1 = 0.0, temp2 = 0.0;
                            for (int jj = 0; jj < N; jj++)
                            {
                                temp1 = temp1 + param->invK[ii + jj*N] * param->y[jj];
                                temp2 = temp2 + param->invK[ii + jj*N] * param->kvec[jj];
                            }
                            muxdx = muxdx - (x[dd] - xmat[ii + dd*N])* param->kvec[ii] * temp1;
                            sigmaxdx = sigmaxdx - (x[dd] - xmat[ii + dd*N])* param->kvec[ii] * temp2;
                        }
                        sigmaxdx = -1*sigmaxdx/sigmax;
                        muxdx = (muxdx-z*sigmaxdx)/sigmax;
                        grad[dd] = ((z*Phiz+phiz)*sigmaxdx + sigmax*Phiz*muxdx)/ (param->koptions.scale*param->koptions.scale);
                    }
                    
                    /*printf("\ngrad: ");
                     * print_vector(nDim, grad);
                     * printf("\nx: ");
                     * print_vector(nDim, (double*)x);*/
                }
                else
                {
                    for (int dd = 0; dd < nDim; dd++)
                        grad[dd] = 0.0;
                }
        }
    }
    return gVal;
}


void recommendSample(double *maxf, double *xnew, double *x, double *y, double ymax, int N, double *kvec, double *invK, kernelOptions koptions, bayesOptions boptions)
{
    
    acqFuncData param;
    param.x = x;
    param.y = y;
    param.invK = invK;
    param.N = N;
    param.kvec = kvec;
    param.koptions = koptions;
    param.boptions = boptions;
    
    param.ymax = ymax;
    param.onlykernel = 0;
    
    void *myacqFuncData = static_cast<void*>(&param);
    
    
    nlopt_opt opt;
    int info;
    
    if (boptions.if_global)
    {
        opt = nlopt_create(NLOPT_GN_ORIG_DIRECT, nDim);// NLOPT_GN_ESCH, NLOPT_GN_ISRES, NLOPT_GN_CRS2_LM, NLOPT_GN_ORIG_DIRECT_L
        nlopt_set_maxtime(opt, OptTime*nDim);
        nlopt_set_lower_bounds(opt, boptions.lb);
        nlopt_set_upper_bounds(opt, boptions.ub);
        nlopt_set_max_objective(opt, acqFunc, myacqFuncData);
        nlopt_set_xtol_rel(opt, 1e-4);
        info = nlopt_optimize(opt, xnew, maxf);
        if (info < 0 || info == 6)
            printf("DIRECT returned with error: %d\n", info);
        /*if (info < 0 || info == 6) {
         * if (info < 0 || info == 6) {
         * printf("Direct failed (%d): trying Nelder-Mead...\n", info);
         * opt = nlopt_create(NLOPT_LN_NELDERMEAD, nDim);
         * nlopt_set_maxtime(opt, OptTime*nDim);
         * nlopt_set_lower_bounds(opt, boptions.lb);
         * nlopt_set_upper_bounds(opt, boptions.ub);
         * nlopt_set_max_objective(opt, acqFunc, myacqFuncData);
         * nlopt_set_xtol_rel(opt, 1e-4);
         * info = nlopt_optimize(opt, xnew, maxf);
         * printf("Nelder-Mead completed (%d)\n", info);
         * }
         * }*/
    }
    else
    {
        opt = nlopt_create(NLOPT_LD_TNEWTON , nDim);// NLOPT_LD_LBFGS, NLOPT_LN_BOBYQA, NLOPT_LD_MMA
        nlopt_set_maxtime(opt, OptTime*nDim);
        nlopt_set_lower_bounds(opt, boptions.lb);
        nlopt_set_upper_bounds(opt, boptions.ub);
        nlopt_set_max_objective(opt, acqFunc, myacqFuncData);
        nlopt_set_xtol_rel(opt, 1e-4);
        nlopt_remove_inequality_constraints(opt);
        nlopt_set_vector_storage(opt, 0);
        info = nlopt_optimize(opt, xnew, maxf);
        /*if (info < 0 || info == 6)
         * printf("BOBYQA returned with error: %d\n", info);*/
    }
}



double distv(double *x1, double *x2)
{
    double sum = 0;
    for (int ii = 0; ii < nDim; ii++)
        sum = sum + (x1[ii] - x2[ii]) * (x1[ii] - x2[ii]);
    return sum;
}

void print_vector(int n, double* v) {
    int j;
    char *ch;
    ch = new char[50];
    for (j = 0; j < n; j++)
    {
        sprintf(ch, " %3.3g, ", v[j]);
        mexPrintf(ch);
    }
    mexPrintf("\n");
}

void print_matrix(int m, int n, double** v) {
    int i, j;
    char *ch;
    ch = new char[100];
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            sprintf(ch, " %3.3g", v[i][j]);
            mexPrintf(ch);
        }
        mexPrintf("\n");
    }
}

