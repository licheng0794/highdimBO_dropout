function [maxf,xnew, nIter] = recommendSampleHighDim(xinit,kvec, gp,method_index,nIter,OptTime)

boptions = gp.boptions;
if(method_index==3) % global optimization
    ymax = max(gp.y);
    maxf = 0;
    [maxf, xnew] = recommendSample(gp.X,gp.y,ymax,gp.N,kvec,gp.invK,...
        boptions.bounds(:,1),boptions.bounds(:,2),gp.M,OptTime,gp.kernelTypeNum,gp.param.kernelScale,...
        gp.param.kernelVar,gp.param.rqalpha,boptions.eps,gp.msrSigma2scalar,boptions.criteriaNum,...
        xinit,1);
    %fprintf('Global maxf: %f\n',maxf);
end