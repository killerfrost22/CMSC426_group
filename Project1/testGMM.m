%% testGMM Method

function cluster = testGMM(model, mu, sigma, tol, k)

    [rows, cols, ~] = size(model);
    final = zeros(rows,cols);
    
    for i = 1:k
         
    for row=1:rows
        for col=1:cols
            x = double(model(row,col,:));
            q = reshape(x,1,[]);
            x = q.';

            if (.5*mvnpdf(x,mu(i,:)',sigma(:,:,i)) >= tol)
                final(row,col) = 1;
            end
        end
    end

    end
        
    cluster = final;
    
end