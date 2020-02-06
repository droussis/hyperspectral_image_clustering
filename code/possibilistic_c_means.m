function [theta,bel,J]=possibilistic_c_means(X,theta)

[l,N]=size(X);
[l,m]=size(theta);
eta=ones(m,1);
e=1;
iter=0;
e_thres=0.001;
max_iter=100;
U=zeros(N,m);
dist=zeros(N,m);
while(e>e_thres && iter<max_iter)
    iter=iter+1;
    for i=1:N
        for j=1:m
            p=X(:,i);
            q=theta(:,j);
            
            % Squared Euclidean Distance
            dist(i,j)=norm(p-q)^2;
        
            % Canberra Distance
            %dist(i,j)=sum((abs(p-q)./(abs(p)+abs(q))));           
            
            U(i,j)=exp(-dist(i,j)/eta(j));
        end
    end
    
    for i=1:N
        [q1,bel]=max(U');
    end

    theta_old=theta;
    theta=zeros(l,m);
    J=0;
    
    for j=1:m
        temp=0;
        for i=1:N
            theta(:,j)=theta(:,j)+U(i,j)*X(:,i);
            eta(j)=eta(j)+U(i,j)*dist(i,j);
            temp=temp+U(i,j);
        end
        theta(:,j)=theta(:,j)/temp;
        J=J+eta(j)+(eta(j)/temp)*(temp*log(temp)-temp);
        eta(j)=eta(j)/temp;
    end

    e=sum(sum(abs(theta-theta_old)));
    
end