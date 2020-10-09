function [ y ] = myFunc( X )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% y = zeros(size(X,1),1);
% for ii = 1:size(X,1)
%     for d = 1:size(X,2)
%         y(ii) = y(ii) + (sum(X(ii,1:d))^2);
%     end
% end

% if we want to get the minimal value, we can use GP-LCB in
% recommendSample.cpp and then re-compile mex file
nDim = size(X,2);
muVec = 2*ones(1,nDim);
SigmaMat = 1*eye(nDim);
cost = mvnpdf(X,muVec,SigmaMat);
y1 = cost*sqrt((2*pi)^nDim*det(SigmaMat));

muVec = 3*ones(1,nDim);
SigmaMat = 1*eye(nDim);
cost = mvnpdf(X,muVec,SigmaMat);
y2 = cost*sqrt((2*pi)^nDim*det(SigmaMat));

y = y1 + 1/2*y2;
end

