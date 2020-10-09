function [K] = computeKernel(X1,X2,param)

scale = param.kernelScale;
kvar = param.kernelVar;
distMat2 = size(size(X1,1),size(X2,1));
if(strcmp(param.kernelType,'SE'))
    if (sum(abs(size(X1)-size(X2)))==0)
        distMat2 = squareform((pdist(X1)).^2);
    else
        for ii = 1 : size(X1,1)
            for jj = 1 : size(X2,1)
                eTemp = X1(ii,:)-X2(jj,:);
                distMat2(ii,jj) = eTemp*eTemp';
            end
        end
    end
    K = kvar*exp(-(1/(2*scale^2))*(distMat2));

elseif(strcmp(param.kernelType,'RQ'))
    alpha = param.rqalpha;
    if (sum(abs(size(X1)-size(X2)))==0)
        distMat2 = squareform((pdist(X1)).^2);
    else
        for ii = 1 : size(X1,1)
            for jj = 1 : size(X2,1)
                eTemp = X1(ii,:)-X2(jj,:);
                distMat2(ii,jj) = eTemp*eTemp';
            end
        end
    end
    K = kvar*(1+(1/(2*alpha*scale^2))*(distMat2)).^(-alpha);
else
    fprintf('Kernel not found!\n');
end
