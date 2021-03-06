function [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha)

classes = unique(Yl);
n_class = length(classes);

for i=1:ntrees,
%     FeaturesToSample(i,:) = randsample(1:size(Xl,2), round(sqrt(size(Xl,2))));
%     PointsToSample(i,:) = randsample(1:size(Xl,1), size(Xl,1), true);
    PointsToSample(i,:) = 1:size(Xl,1);
    trees{i} = ClassificationTree.fit(Xl(PointsToSample(i,:),:),Yl(PointsToSample(i,:)), 'NVarToSample', round(sqrt(size(Xl,2))),'Prune','off');
    
    [Ytl(i,:), Ptl(i,:,:)] = predict(trees{i}, Xl);
    [Ytu(i,:), Ptu(i,:,:)] = predict(trees{i}, Xu);
end

[Yfl, Pl] = forest_response(Ytl, Ptl, classes);
[Yfu, Pu] = forest_response(Ytu, Ptu, classes);

fprintf('Initial accuracy on labeled: %f \n', sum(Yfl==Yl)/length(Yl));
acc = sum(Yfu==Yu)/length(Yu);
fprintf('Initial accuracy on unlabeled: %f \n', acc);

for m = 1:epochs,
    
    T = T0*exp(-(m-1)/tau);     %reduce temp value
    
    %margin max loss fcn
    for c = 1:n_class,
        lg(:,c) = exp(-2*(Pu(:,c)-max(Pu(:,classes(classes ~= c)),[],2)));
    end
    
    %compute optimal distribution for drawing labels
    Pu_opt = exp(-(alpha*lg+T)/T);
    Z = sum(Pu_opt,2); Z = repmat(Z,[1 n_class]);
    Pu_opt = Pu_opt./Z;
    
    %draw random labels from Pu_opt for each unlabeled data point and each tree
    for samp = 1:size(Xu,1),
        Yu_est(samp, :) = randsample(1:n_class, ntrees, true, Pu_opt(samp,:));
    end
    
    %training the new forest on both labeled on unlabeled data
    Xtotal = [Xl; Xu];
    clear trees FeaturesToSample PointsToSample Ytl Ytu Ptl Ptu;
    for i=1:ntrees,
        Ytotal = [Yl; Yu_est(:,i)];
        %         FeaturesToSample(i,:) = randsample(1:size(Xtotal,2), round(sqrt(size(Xtotal,2))));
        PointsToSample(i,:) = randsample(1:size(Xtotal,1), size(Xtotal,1), true);
        %PointsToSample(i,:) = 1:size(Xtotal,1);
        trees{i} = ClassificationTree.fit(Xtotal(PointsToSample(i,:),:),Ytotal(PointsToSample(i,:)), 'NVarToSample', round(sqrt(size(Xtotal,2))),'Prune','off');
        
        [Ytl(i,:), Ptl(i,:,:)] = predict(trees{i}, Xl);
        [Ytu(i,:), Ptu(i,:,:)] = predict(trees{i}, Xu);
    end
    
    [Yfl, Pl] = forest_response(Ytl, Ptl, classes);
    [Yfu, Pu] = forest_response(Ytu, Ptu, classes);

%     fprintf('Accuracy on labeled: %f \n', sum(Yfl==Yl)/length(Yl));
    acc = [acc, sum(Yfu==Yu)/length(Yu)];
    fprintf('Accuracy on unlabeled: %f \n', acc(end));
    
end

end