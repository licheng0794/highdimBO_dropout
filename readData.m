function [data] = readData(gp,xnew)

xnew = xnew.*(gp.max_x - gp.min_x) + gp.min_x;
y = myFunc(xnew);
multp = (pinv(diag(gp.max_x-gp.min_x)));
xnew = (xnew-repmat(gp.min_x,size(xnew,1),1))*multp;

data.X = xnew;
data.y = y;
if isnan(y)
    keyboard
end





