function [theta,bel,J]=fuzzy_c_means(X,theta,fuzz)

[l,N]=size(X);
[l,m]=size(theta);
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
            temp=0;
            p=X(:,i);
            q=theta(:,j);
            
            % Squared Euclidean Distance
            dist_j(i,j)=norm(p-q)^2;
        
            % Canberra Distance
            %dist_j(i,j)=sum((abs(p-q)./(abs(p)+abs(q))));
            
            for k=1:m
                r=theta(:,k);
                
                %Squared Euclidean Distance
                dist_k(i,k)=norm(p-r)^2;
                
                %Canberra Distance
                %dist_k(i,k)=sum((abs(p-r)./(abs(p)+abs(r))));
                
                if dist_j(i,j)~=0 && dist_k(i,k)~=0
                    temp=temp+(dist_j(i,j)/dist_k(i,k));
                end
            end
            U(i,j)=1/(temp^(1/(fuzz-1)));
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
            theta(:,j)=theta(:,j)+U(i,j)^fuzz*X(:,i);
            temp=temp+U(i,j)^fuzz;
        end
        theta(:,j)=theta(:,j)/temp;
        J=J+temp*sum(dist_j(:,j));
    end


    e=sum(sum(abs(theta-theta_old)));
    
end