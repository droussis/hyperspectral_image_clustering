function [rand_index,jaccard_coefficient]=metrics(y_pred,y_real)

a=0;
b=0;
c=0;
d=0;
y_pred=y_pred';
y_real=y_real';
n=size(y_real);

for i=1:n
    for j=i+1:n
        if (y_pred(i)==y_pred(j)) && (y_real(i)==y_real(j))
            a=a+1;
        elseif (y_pred(i)==y_pred(j)) && (y_real(i)~=y_real(j))
            b=b+1;
        elseif (y_pred(i)~=y_pred(j)) && (y_real(i)==y_real(j))
            c=c+1;
        elseif (y_pred(i)~=y_pred(j)) && (y_real(i)~=y_real(j))
            d=d+1;
        end
    end
end

rand_index=(a+d)/(a+b+c+d);
jaccard_coefficient=a/(a+b+c);

end
