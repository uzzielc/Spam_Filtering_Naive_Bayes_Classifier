% Original name of file was HW2_2.m

z = dlmread('spambase.data',',');
rng(0); % initialize the random number generator
rp = randperm(size(z,1)); % random permutation of indices
z = z(rp,:); % shuffle the rows of z
x = z(:,1:end-1); % feature vectors
y = z(:,end); % labels

% quantize the variables

% find the median and adjust the values in the columns accordingly
for i=1:57
   med = median(x(:,i));
   indices1 = find(x(:,i) < med);
   indices2 = find(x(:,i) > med);
   indices3 = find(x(:,i) == med);
   x(indices1,i) = 1;
   x(indices2,i) = 2;
   x(indices3,i) = 1;
end

n = 2000; % number of elements in training data

% split the data into testing and training sets
train_x = x(1:n,1:end);
train_y = y(1:n,1:end);

test_x = x(n+1:end,1:end);
test_y = y(n+1:end,1:end);


% compute the estimation parameters
n_1 = nnz(train_y);
n_0 = length(train_y) - n_1;

pi_0_hat = n_0/n;
pi_1_hat = n_1/n;


N = zeros(2,2,57); % N will hold the values n_kl^(j) at N(k+1,l,j)
% for our problem, k is either 0,1 and l is either 1,2

for j = 1:57 % calculate the values for n_kl^(j), i.e. count occurences according to conditions
    for i = 1:n
        if y(i) == 0 && train_x(i,j) == 1 % if y(i) = k = 0 & x(i,j) = l = 1
            N(1,1,j) = 1 + N(1,1,j);
        
        elseif y(i) == 1 && train_x(i,j) == 1
            N(2,1,j) = 1 + N(2,1,j);
            
        elseif y(i) == 0 && train_x(i,j) == 2
            N(1,2,j) = 1 + N(1,2,j);
            
        elseif y(i) == 1 && train_x(i,j) == 2
            N(2,2,j) = 1 + N(2,2,j);
        end 
    end
end

% calculate individiual error for each feature vector in the test set.
for i = 1:size(test_y)
    p0 = P_0_hat(test_x(i,:),N,n_0,pi_0_hat); % calc prob that label is 0
    
    p1 = P_1_hat(test_x(i,:),N,n_1,pi_1_hat); % calc prob that label is 1
    P(i) = max(p0,p1);
    if P(i) == p0
        test_label(i) = 0;
    else
        test_label(i) = 1;
    end
    check(i) = test_label(i) - test_y(i);
    test_error = nnz(check)/size(test_y,1);
end


function P = P_0_hat(X,N,n_0,pi_0_hat)
    for j = 1:57
        if j == 1
            P =  N(1,X(j),j)/(N(1,1,j) + N(1,2,j));
        else
            P = P * N(1,X(j),j)/(N(1,1,j) + N(1,2,j));
        end
    end
    P = pi_0_hat * P;
end

function P = P_1_hat(X,N,n_1,pi_1_hat)
    for j = 1:57
        if j == 1
            P =  N(2,X(j),j)/(N(2,1,j) + N(2,2,j));
        else
            P = P * N(2,X(j),j)/((N(2,1,j) + N(2,2,j)));
        end
    end
    P = pi_1_hat * P;
end
