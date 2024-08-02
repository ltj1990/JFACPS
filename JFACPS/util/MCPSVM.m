function predict_label = MCPSVM(Xs, Ys, Xt, options)

    eta = options.eta;
    t1 = options.t1;
    t2 = options.t2;
    ind1 = find(Ys==1);
    x1data = Xs(ind1,:);
    ind2 = find(Ys==2);
    x2data = Xs(ind2,:);
    G = [x1data ones(size(x1data,1),1)]'* [x1data ones(size(x1data,1),1)];
    G1 = G + eta*eye(size(G));
    H = [x2data ones(size(x2data,1),1)]'* [x2data ones(size(x2data,1),1)];
    H1 = H + eta*eye(size(H));
    
    % Model solving
    M1 = CS_GeometricMean(inv(G1),H1,t1);
    M2 = CS_GeometricMean(inv(H1),G1,t2);
    Z1 = chol(M1)';
    Z2 = chol(M2)';
    error = 0;
    for l = 1:size(Xt,1)
        dist1 = norm([Xt(l,:) 1]*Z1);
        dist2 = norm([Xt(l,:) 1]*Z2);
        if (dist1<dist2)
            predict_label(l) = 1;
        else
            predict_label(l) = 2;
        end
    end
    predict_label = predict_label';
end


