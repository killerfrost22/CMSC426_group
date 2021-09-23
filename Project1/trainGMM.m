function [mu, sigma] = trainGMM(x, k)
    % ARGS
    % x is points
    % k is # of gaussians
    num_points = size(x);

    % set convergence criteria
    thresh = .001;
    iter = 0;
    max_i = 100;

    % initialize model
    mean = zeros(k, 3);
    prev_mu = mean;
    covariance = zeros(3, 3, k);
    pi = zeros(k,1);
    for i=1:k
        for j=1:3
            mean(i, j) = 1+ 253*rand;
        end
    end
    for i=1:3
        for j=1:3
            for l=1:k
                covariance(1,1,l) = 1000;
                covariance(2,2,l) = 1000;
                covariance(3,3,l) = 1000;
            end
        end
    end
    for i=1:k
        pi(i) = 100*rand;
    end
    cluster_weight = zeros(num_points(1), k);
    % while loop (training)
    while (iter < max_i && abs(sum(sum(prev_mu - mean))) > thresh)
        prev_mu = mean;
        % EXPECTATION
        for j = 1:k
            cluster_weight(:,j) = 1e-10 + pi(j)*mvnpdf(x, mean(j,:), covariance(:,:,j));
        end
        for i=1:num_points(1)
            sum_=0;
            for j=1:k
                sum_ = sum_ + sum(cluster_weight(i,j));
            end
            cluster_weight(i,:) = cluster_weight(i,:) ./ sum_;
        end
        %cluster_weight = cluster_weight / sum_;
        % MAXIMIZATION
        % mean
        for color=1:3
            for j=1:k
                top_sum = 0;
                bot_sum = 0;
                for i=1:num_points(1)
                    top_sum = top_sum + cluster_weight(i,j) * x(i,color);
                    bot_sum = bot_sum + cluster_weight(i,j);
                end
                mean(j, color) = top_sum / bot_sum;
            end
        end
        % cov
        for j=1:k
            top_sum = zeros(3, 3);
            for i=1:num_points(1)
                top_sum = top_sum + cluster_weight(i,j) * (((x(i,:) - mean(j,:))') * (x(i,:) - mean(j,:)));
            end
            bot_sum = sum(cluster_weight(:,j));
            covariance(:,:,j) = top_sum / bot_sum;
        end
        % pi
        for j=1:k
            pi(j) = sum(cluster_weight(:,j)) / num_points(1);
        end
        % increment i
        iter = iter + 1;

    end 
    % save model & end procedure
    mu = mean;
    sigma = covariance;
    pi_ = pi;


end