
% author: Cheng Li, lichengxlxl@gmail.com
clear

%% initial the number of data nDim + 1
rng(1)
nDim = 5; 
nBandits = 2;
N0= nDim + 1;
% a small tip: if the dimension is very high, the initial points are
% expected to be a bit high. We can do it by using a norrow Initbounds
Initbounds = repmat([1 4],nDim,1); % the space of the initial points 

bounds = repmat([1 4],nDim,1); % the whole search space
data.max_x = max(bounds,[],2)';
data.min_x = min(bounds,[],2)';

% because we used lengthscale=0.1 in GP, we normalize the search space to
% [0 1]
boptions.bounds = (bounds - repmat(data.min_x',1,2))./repmat(data.max_x'-data.min_x',1,2);  
multp = (pinv(diag(data.max_x-data.min_x)));

%% set the parameters of optimizer
boptions.criteria = 'gpUCB'; %'EI', 'PI','gpUCB'
if strcmp(boptions.criteria,'gpUCB')
    criteriaNum = 3;
elseif strcmp(boptions.criteria,'EI')
    criteriaNum = 1;
elseif strcmp(boptions.criteria,'PI')
    criteriaNum = 2;
else
    criteriaNum = 1;
end
boptions.criteriaNum = criteriaNum;
boptions.eps = 0.0;
boptions.optMethod = 'Continuous';
boptions.nBandits = nBandits;

MaxIter = 100;
MaxRun = 2;
boptions.Initbounds = Initbounds; % the space of the initial data
% boptions.bounds = repmat([-1 1],nDim,1); % the whole search space

%% the setting of running algorithms
if_DropOut_random = 1; % DropOut_random
if_DropOut_copy = 1; % DropOut_copy
if_DropOut_mix = 1;  % DropOut_mix
if_Global = 1;


yG = zeros(MaxRun,MaxIter);

yDropoutRand = zeros(MaxRun, MaxIter);
yDropoutCopy = zeros(MaxRun, MaxIter);
yDropoutMix = zeros(MaxRun, MaxIter);


for mmRun = 1:MaxRun
    fprintf('+++++++++nDim:%d, Runs:%d+++++++++\n', nDim, mmRun);
    
    % initialize data
    X = repmat(Initbounds(:,1)',N0,1)+repmat([Initbounds(:,2)-Initbounds(:,1)]',N0,1).*rand(N0,nDim);
    y = myFunc(X); % the initial X and y
    X = (X-repmat(data.min_x,size(X,1),1))*multp;
    data.X = X;
    
    % standardization y
    data.max_y = max(y);
    data.min_y = min(y);
    data.mean_y = mean(y);
    data.var_y = var(y);
    y = (y-data.mean_y)/sqrt(data.var_y);
    data.y = y;
    
    mainScript_BayesOptHighDim % the main part is running inside
    if if_Global
        yG(mmRun,:) = gpTarget.y'*sqrt(gpTarget.var_y) + gpTarget.mean_y;
    end
    if if_DropOut_random
        yDropoutRand(mmRun,:) = gpTarget_DropRand.y'*sqrt(gpTarget_DropRand.var_y) + gpTarget_DropRand.mean_y;
    end
    if if_DropOut_copy
        yDropoutCopy(mmRun,:) = gpTarget_DropCopy.y'*sqrt(gpTarget_DropCopy.var_y) + gpTarget_DropCopy.mean_y;
    end
    if if_DropOut_mix
        yDropoutMix(mmRun,:) = gpTarget_DropMix.y'*sqrt(gpTarget_DropMix.var_y) + gpTarget_DropMix.mean_y;
    end
    
end

figure;

yGm = zeros(MaxRun,MaxIter);
yDropoutRandM = zeros(MaxRun, MaxIter);
yDropoutCopyM = zeros(MaxRun, MaxIter);
yDropoutMixM = zeros(MaxRun, MaxIter);

if if_Global
    for ii = 1:MaxRun
        for jj = 1:MaxIter
            yGm(ii,jj) = max(yG(ii,1:jj));
        end
    end
    errorbar(mean(yGm,1),std(yGm,1)/sqrt(MaxRun),'-^m');
    hold on;
end

if if_DropOut_random
    for ii = 1:MaxRun
        for jj = 1:MaxIter
            yDropoutRandM(ii,jj) = max(yDropoutRand(ii,1:jj));
        end
    end
    errorbar(mean(yDropoutRandM,1),std(yDropoutRandM,1)/sqrt(MaxRun),'-og');
    hold on;
end

if if_DropOut_copy
    for ii = 1:MaxRun
        for jj = 1:MaxIter
            yDropoutCopyM(ii,jj) = max(yDropoutCopy(ii,1:jj));
        end
    end
    errorbar(mean(yDropoutCopyM,1),std(yDropoutCopyM,1)/sqrt(MaxRun),'-sb');
    hold on;
end

if if_DropOut_mix
    for ii = 1:MaxRun
        for jj = 1:MaxIter
            yDropoutMixM(ii,jj) = max(yDropoutMix(ii,1:jj));
        end
    end
    errorbar(mean(yDropoutMixM,1),std(yDropoutMixM,1)/sqrt(MaxRun),'->r');
    hold on;
end


legend('DIRECT','Dropout-Random','Dropout-Copy', 'Dropout-Mix');








