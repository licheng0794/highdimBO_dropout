function gp = buildGP(data,param,boptions)

K = computeKernel(data.X,data.X,param);

gp.K = K;
gp.X = data.X;
gp.y = data.y;
if(length(param.msrSigma2)<2)
    gp.msrSigma2 = param.msrSigma2*ones(size(K,1),1);
else
    gp.msrSigma2 = param.msrSigma2;
end
gp.msrSigma2scalar = mean(param.msrSigma2);
gp.invK = inv(K+diag(gp.msrSigma2)); 
gp.param = param;
gp.max_x = data.max_x;
gp.min_x = data.min_x;

gp.mean_y = data.mean_y;
gp.var_y = data.var_y;
gp.N = size(gp.X,1);
gp.M = size(gp.X,2);
gp.kernelTypeNum = param.kernelTypeNum;
gp.boptions = boptions;
