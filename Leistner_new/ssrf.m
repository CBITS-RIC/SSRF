function [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha)

classes = unique(Yl);
n_class = length(classes);

for i=1:ntrees,
    sampled_features(i,:) = randsample(1:size(Xl,2), round(sqrt(size(Xl,2))));
    sampled_points(i,:) = randsample(1:size(Xl,1), size(Xl,1), true);
    trees{i} = ClassificationTree.fit(Xl(sampled_points(i,:),sampled_features(i,:)),Yl(sampled_points(i,:)));
    Ytl(i,:) = predict(trees{i}, Xl(:,sampled_features(i,:)));
    Ytu(i,:) = predict(trees{i}, Xu(:,sampled_features(i,:)));
end

[Yfl, Pl] = forest_response(Ytl, classes);
[Yfu, Pu] = forest_response(Ytu, classes);
fprintf('Accuracy on labeled: %f \n', sum(Yfl==Yl)/length(Yl));
acc = sum(Yfu==Yu)/length(Yu);
fprintf('Accuracy on unlabeled: %f \n', acc);

lgs = [];

for m = 1:epochs,
    
    T = T0*exp(-(m-1)/tau);     %reduce temp value
    Tvals(m) = T;               %save T values
    
    %margin max loss fcn (Entropy)
    for c = 1:n_class
        lg(:,c) = exp(-2*(Pu(:,c)-max(Pu(:,classes(classes ~= c)),[],2)));
        lgs = [lgs; mean(mean(lg))];
    end
    
    %Compute Optimal Distribution over predicted labels
    Pu_opt = exp(-(alpha*lg+T)/T);
    %                 Pu_opt = max(Pu_opt,eps);       %prevents Nan if Pu_opt is 0
    Z = sum(Pu_opt,2); Z = repmat(Z,[1 n_class]);
    Pu_opt = Pu_opt./Z; %normalized probabilities
    
    %draw random labels from Pu_opt distribution for each unlabeled data point and each tree
    for samp = 1:size(Xu,1);
        Yu_est(samp, :) = randsample(1:n_class, ntrees, true, Pu_opt(samp,:));
    end
    
    %training the new forest
    Xtotal = [Xl; Xu];
    clear trees sampled_features sampled_points Ytl Ytu;
    for i=1:ntrees,
        Ytotal = [Yl; Yu_est(:,i)];
        sampled_features(i,:) = randsample(1:size(Xtotal,2), round(sqrt(size(Xtotal,2))));
        sampled_points(i,:) = randsample(1:size(Xtotal,1), size(Xtotal,1), true);
        trees{i} = ClassificationTree.fit(Xtotal(sampled_points(i,:),sampled_features(i,:)),Ytotal(sampled_points(i,:)));
        Ytl(i,:) = predict(trees{i}, Xl(:,sampled_features(i,:)));
        Ytu(i,:) = predict(trees{i}, Xu(:,sampled_features(i,:)));
    end
    
    [Yfl Pl] = forest_response(Ytl, classes);
    [Yfu Pu] = forest_response(Ytu, classes);
%     fprintf('Accuracy on labeled: %f \n', sum(Yfl==Yl)/length(Yl));
    acc = [acc, sum(Yfu==Yu)/length(Yu)];
    fprintf('Accuracy on unlabeled: %f \n', acc(end));
    
end

fprintf('\n');


end