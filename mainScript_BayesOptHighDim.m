% clear;


%% Load Source and Target Data
dataTarget = data;

%% Set the kernel type
kernelType = 'SE';%'RQ', 'SE'
if strcmp(kernelType,'SE')
    kernelTypeNum = 1;
elseif strcmp(kernelType,'RQ')
    kernelTypeNum = 2;
else
    kernelTypeNum = 1;
end

paramTarget.kernelType = kernelType;
paramTarget.kernelVar = 1;
paramTarget.rqalpha = 10;
paramTarget.msrSigma2 = 0.01; % noise
paramTarget.kernelTypeNum = kernelTypeNum;
paramTarget.kernelScale = 0.1; % the lengthscale for SE kernel

%% Build Target GP original
gpTarget = buildGP(dataTarget,paramTarget,boptions);
gpTarget.yplot = gpTarget.y;


gpTarget_DropRand = gpTarget;
gpTarget_DropCopy = gpTarget;
gpTarget_DropMix = gpTarget;

%% Bayesian Optimization loop
GetX = 1;
iter = length(gpTarget.y);

delta_gpucb = 0.1; % it is used to setup the tradeoff parameter in GP-UCB
nBandits = boptions.nBandits;
iter_t = 0;
while(GetX)
    iter = iter + 1;
    iter_t = iter_t +1;
    fprintf('======iter %d =======\n',iter);
    
    
    %% HighDimension recommendation
    nIter = 10;
    OptTime = 30; % the total optimization time for each global optimization
    %% Multistart
    
    if if_Global
        
        xinit = rand(gpTarget.M,1);
        kvec = zeros(gpTarget.N,1);
        d = gpTarget.M;
        boptions.eps = 2*log(iter_t^2*2*pi^2/(3*delta_gpucb)) + 2*d*(log(d*iter_t^2)+1/2*log(log(4*d/delta_gpucb)));
        gpTarget.boptions.eps = boptions.eps;
        [maxfG, xnewG] = recommendSampleHighDim(xinit, kvec, gpTarget, 3, nIter, OptTime/gpTarget.M);

        % If it is too close, perturb it a bit
        if min( sqrt( sum( bsxfun(@minus, gpTarget.X, xnewG).^2, 2) ))...
                < 1e-10
            while min( sqrt( sum( bsxfun(@minus, gpTarget.X, xnewG).^2,2)))...
                    < 1e-10
                xnewG = projectToRectangle( ...
                    xnewG' + 0.01 * randn(gpTarget.M, 1), gpTarget.boptions.bounds)';
            end
        end
        
        newdata = readData(gpTarget,xnewG);
        data.X = [data.X; newdata.X]; % X can be added directly
        data.y = [data.y*sqrt(data.var_y) + data.mean_y; newdata.y]; % Y need to re-standardize
        data.max_y = max(data.y);
        data.min_y = min(data.y);
        data.mean_y = mean(data.y);
        data.var_y = var(data.y);
        y = (data.y-data.mean_y)/sqrt(data.var_y);
        data.y = y;
        gpTarget = buildGP(data,paramTarget,boptions);
        ymax = max(gpTarget.y);
        fprintf('Global    : ymax: %f, ycurrent: %f\n',ymax*sqrt(gpTarget.var_y)+gpTarget.mean_y,gpTarget.y(end)*sqrt(gpTarget.var_y)+gpTarget.mean_y);
    end
    
    
    if if_DropOut_random
        % dropout_random
        kvec = zeros(gpTarget_DropRand.N,1);
        subIdx = randperm(size(gpTarget_DropRand.X,2), nBandits);
        restIdx = setdiff(1:size(gpTarget_DropRand.X,2), subIdx);
        % rebuild GP_subhighdim
        dataTarget_sub.mean_y = gpTarget_DropRand.mean_y;
        dataTarget_sub.var_y = gpTarget_DropRand.var_y;
        dataTarget_sub.y = gpTarget_DropRand.y;
        dataTarget_sub.min_x = gpTarget_DropRand.min_x(1,subIdx);
        dataTarget_sub.max_x = gpTarget_DropRand.max_x(1,subIdx);
        dataTarget_sub.X = gpTarget_DropRand.X(:,subIdx);
        d = nBandits;
        boptions.eps = 2*log(iter_t^2*2*pi^2/(3*delta_gpucb)) + 2*d*(log(d*iter_t^2)+1/2*log(log(4*d/delta_gpucb)));
        boptions_sub = boptions;
        boptions_sub.Initbounds = boptions.Initbounds(subIdx,:);
        boptions_sub.bounds = boptions.bounds(subIdx,:);
        gpTarget_sub = buildGP(dataTarget_sub,paramTarget,boptions_sub);
        initx = rand(length(subIdx),1);
        [maxsub, xnewM_sub] = recommendSampleHighDim(initx, kvec, gpTarget_sub, 3, nIter, OptTime./(gpTarget_sub.M));
        xnewM_rest = rand(1, length(restIdx));
        temp = [xnewM_sub xnewM_rest];
        tempIdx = [subIdx restIdx];
        [~,I] = sort(tempIdx, 2, 'ascend');
        xnewHM = temp(I);  %
        
        newdata = readData(gpTarget_DropRand,xnewHM);
        gpTarget_DropRand.X = [gpTarget_DropRand.X; newdata.X]; % X can be added directly
        gpTarget_DropRand.y = [gpTarget_DropRand.y*sqrt(gpTarget_DropRand.var_y) + gpTarget_DropRand.mean_y; newdata.y]; % Y need to re-standardize
        gpTarget_DropRand.max_y = max(gpTarget_DropRand.y);
        gpTarget_DropRand.min_y = min(gpTarget_DropRand.y);
        gpTarget_DropRand.mean_y = mean(gpTarget_DropRand.y);
        gpTarget_DropRand.var_y = var(gpTarget_DropRand.y);
        y = (gpTarget_DropRand.y-gpTarget_DropRand.mean_y)/sqrt(gpTarget_DropRand.var_y);
        gpTarget_DropRand.y = y;
        gpTarget_DropRand.N = gpTarget_DropRand.N+1;
        gpTarget_DropRand.msrSigma2 = [gpTarget_DropRand.msrSigma2;gpTarget_DropRand.msrSigma2scalar];
        % here we do not need to rebuild GP
        ymax = max(gpTarget_DropRand.y);
        fprintf('Dropout Random Dims %d: ymax: %f, ycurrent: %f\n',nBandits, ymax*sqrt(gpTarget_DropRand.var_y)+gpTarget_DropRand.mean_y,gpTarget_DropRand.y(end)*sqrt(gpTarget_DropRand.var_y)+gpTarget_DropRand.mean_y);
        
        
    end
    
    
    if if_DropOut_copy
        % dropout_copy
        kvec = zeros(gpTarget_DropCopy.N,1);
        subIdx = randperm(size(gpTarget_DropCopy.X,2), nBandits); 
        restIdx = setdiff(1:size(gpTarget_DropCopy.X,2), subIdx);
        % rebuild GP_subhighdim
        dataTarget_sub.mean_y = gpTarget_DropCopy.mean_y;
        dataTarget_sub.var_y = gpTarget_DropCopy.var_y;
        dataTarget_sub.y = gpTarget_DropCopy.y;
        dataTarget_sub.min_x = gpTarget_DropCopy.min_x(1,subIdx);
        dataTarget_sub.max_x = gpTarget_DropCopy.max_x(1,subIdx);
        dataTarget_sub.X = gpTarget_DropCopy.X(:,subIdx);
        
        d = nBandits;
        boptions.eps = 2*log(iter_t^2*2*pi^2/(3*delta_gpucb)) + 2*d*(log(d*iter_t^2)+1/2*log(log(4*d/delta_gpucb)));
        boptions_sub = boptions;
        boptions_sub.Initbounds = boptions.Initbounds(subIdx,:);
        boptions_sub.bounds = boptions.bounds(subIdx,:);
        gpTarget_sub = buildGP(dataTarget_sub,paramTarget,boptions_sub);
        initx = rand(length(subIdx),1);
        
        [maxsub, xnewM_sub] = recommendSampleHighDim(initx, kvec, gpTarget_sub, 3, nIter, OptTime./(gpTarget_sub.M));
        %     xnewM_rest = rand(1, length(restIdx));
        maxIdx = find(gpTarget_DropCopy.y ==max(gpTarget_DropCopy.y));
        xnewM_rest = gpTarget_DropCopy.X(maxIdx(1), restIdx);
        temp = [xnewM_sub xnewM_rest];
        tempIdx = [subIdx restIdx];
        [~,I] = sort(tempIdx, 2, 'ascend');
        xnewHM = temp(I);  %
        
        newdata = readData(gpTarget_DropCopy,xnewHM);
        gpTarget_DropCopy.X = [gpTarget_DropCopy.X; newdata.X]; % X can be added directly
        gpTarget_DropCopy.y = [gpTarget_DropCopy.y*sqrt(gpTarget_DropCopy.var_y) + gpTarget_DropCopy.mean_y; newdata.y]; % Y need to re-standardize
        gpTarget_DropCopy.max_y = max(gpTarget_DropCopy.y);
        gpTarget_DropCopy.min_y = min(gpTarget_DropCopy.y);
        gpTarget_DropCopy.mean_y = mean(gpTarget_DropCopy.y);
        gpTarget_DropCopy.var_y = var(gpTarget_DropCopy.y);
        y = (gpTarget_DropCopy.y-gpTarget_DropCopy.mean_y)/sqrt(gpTarget_DropCopy.var_y);
        gpTarget_DropCopy.y = y;
        gpTarget_DropCopy.N = gpTarget_DropCopy.N+1;
        gpTarget_DropCopy.msrSigma2 = [gpTarget_DropCopy.msrSigma2;gpTarget_DropCopy.msrSigma2scalar];
        % here we do not need to rebuild GP
        ymax = max(gpTarget_DropCopy.y);
        fprintf('Dropout Copy Dims %d: ymax: %f, ycurrent: %f\n',nBandits, ymax*sqrt(gpTarget_DropCopy.var_y)+gpTarget_DropCopy.mean_y,gpTarget_DropCopy.y(end)*sqrt(gpTarget_DropCopy.var_y)+gpTarget_DropCopy.mean_y);
        
    end
    
    if if_DropOut_mix
        % dropout_mix     
        kvec = zeros(gpTarget_DropMix.N,1);
        subIdx = randperm(size(gpTarget_DropMix.X,2), nBandits);
        restIdx = setdiff(1:size(gpTarget_DropMix.X,2), subIdx);
        % rebuild GP_Beta_sub
        dataTarget_sub.mean_y = gpTarget_DropMix.mean_y;
        dataTarget_sub.var_y = gpTarget_DropMix.var_y;
        dataTarget_sub.y = gpTarget_DropMix.y;
        dataTarget_sub.min_x = gpTarget_DropMix.min_x(1,subIdx);
        dataTarget_sub.max_x = gpTarget_DropMix.max_x(1,subIdx);
        dataTarget_sub.X = gpTarget_DropMix.X(:,subIdx);
        d = nBandits;
        boptions.eps = 2*log(iter_t^2*2*pi^2/(3*delta_gpucb)) + 2*d*(log(d*iter_t^2)+1/2*log(log(4*d/delta_gpucb)));
        boptions_sub = boptions;
        boptions_sub.Initbounds = boptions.Initbounds(subIdx,:);
        boptions_sub.bounds = boptions.bounds(subIdx,:);
        gpTarget_sub = buildGP(dataTarget_sub,paramTarget,boptions_sub);
        

        initx = rand(length(subIdx),1);
        
        [maxsub, xnewM_sub] = recommendSampleHighDim(initx, kvec, gpTarget_sub, 3, nIter, OptTime./(gpTarget_sub.M));
        if rand >= 0.1 % set p = 0.1
            maxIdx = find(gpTarget_DropMix.y ==max(gpTarget_DropMix.y));
            xnewM_rest = gpTarget_DropMix.X(maxIdx(1), restIdx);
        else
            xnewM_rest = rand(1, gpTarget_DropMix.M);
        end
        
        temp = [xnewM_sub xnewM_rest];
        tempIdx = [subIdx restIdx];
        [~,I] = sort(tempIdx, 2, 'ascend');
        xnewHM = temp(I);  %
        
        newdata = readData(gpTarget_DropMix,xnewHM);
        gpTarget_DropMix.X = [gpTarget_DropMix.X; newdata.X]; % X can be added directly
        gpTarget_DropMix.y = [gpTarget_DropMix.y*sqrt(gpTarget_DropMix.var_y) + gpTarget_DropMix.mean_y; newdata.y]; % Y need to re-standardize
        gpTarget_DropMix.max_y = max(gpTarget_DropMix.y );
        gpTarget_DropMix.min_y = min(gpTarget_DropMix.y );
        gpTarget_DropMix.mean_y = mean(gpTarget_DropMix.y );
        gpTarget_DropMix.var_y = var(gpTarget_DropMix.y );
        y = (gpTarget_DropMix.y -gpTarget_DropMix.mean_y)/sqrt(gpTarget_DropMix.var_y);
        gpTarget_DropMix.y = y;
        gpTarget_DropMix.N = gpTarget_DropMix.N+1;
        gpTarget_DropMix.msrSigma2 = [gpTarget_DropMix.msrSigma2;gpTarget_DropMix.msrSigma2scalar];
        % here we do not need to rebuild GP
        ymax = max(gpTarget_DropMix.y);
        fprintf('Dropout Mix Dims %d: ymax: %f, ycurrent: %f\n',nBandits, ymax*sqrt(gpTarget_DropMix.var_y)+gpTarget_DropMix.mean_y,gpTarget_DropMix.y(end)*sqrt(gpTarget_DropMix.var_y)+gpTarget_DropMix.mean_y);
    end
    
    if(iter>=MaxIter)
        break;
    end
end


