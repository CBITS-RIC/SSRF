% Semi-Supervised Stumps (SSS)

function [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha, nvartosample)

classes = unique(Yl);
n_class = length(classes);

chance = 1/length(classes); %output probablity for non-observed classes
nochance = 2/length(classes);   %multiplication factor for observed classes

for i=1:ntrees,
    ClassesToSample = sort(randsample(classes,2));
    ClassesNotToSample = classes;
    ClassesNotToSample(ClassesToSample) = [];
    Xtrain = [Xl(Yl==ClassesToSample(1),:); Xl(Yl==ClassesToSample(2),:)];
    Ytrain = [Yl(Yl==ClassesToSample(1)); Yl(Yl==ClassesToSample(2))];
    PointsToSample = randsample(1:size(Xtrain,1), size(Xtrain,1), true);
    trees{i} = ClassificationTree.fit(Xtrain(PointsToSample,:), Ytrain(PointsToSample), 'NVarToSample', nvartosample, 'MinParent', size(Xtrain,1), 'Prune','off');
    [Ytl(i,:), Ptl(i,:,ClassesToSample)] = predict(trees{i}, Xl);
    [Ytu(i,:), Ptu(i,:,ClassesToSample)] = predict(trees{i}, Xu);
    %adjusting the probabilities for non-observed classes:
    Ptl(i,:,ClassesToSample) = Ptl(i,:,ClassesToSample)*nochance;
    Ptl(i,:,ClassesNotToSample) = chance;
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
    clear trees Ytl Ytu Ptl Ptu;
    for i=1:ntrees,
        Ytotal = [Yl; Yu_est(:,i)];
        ClassesToSample = sort(randsample(classes,2));
        ClassesNotToSample = classes;
        ClassesNotToSample(ClassesToSample) = [];
        Xtrain = [Xtotal(Ytotal==ClassesToSample(1),:);Xtotal(Ytotal==ClassesToSample(2),:)];
        Ytrain = [Ytotal(Ytotal==ClassesToSample(1));Ytotal(Ytotal==ClassesToSample(2))];
        PointsToSample = randsample(1:size(Xtrain,1), size(Xtrain,1), true);
        trees{i} = ClassificationTree.fit(Xtrain(PointsToSample,:), Ytrain(PointsToSample), 'NVarToSample', nvartosample, 'MinParent', size(Xtrain,1), 'Prune','off');
        [Ytl(i,:), Ptl(i,:,ClassesToSample)] = predict(trees{i}, Xl);
        [Ytu(i,:), Ptu(i,:,ClassesToSample)] = predict(trees{i}, Xu);
        %adjusting the probabilities for non-observed classes:
        Ptl(i,:,ClassesToSample) = Ptl(i,:,ClassesToSample)*nochance;
        Ptl(i,:,ClassesNotToSample) = chance;
    end
    
    [Yfl, Pl] = forest_response(Ytl, Ptl, classes);
    [Yfu, Pu] = forest_response(Ytu, Ptu, classes);

%     fprintf('Accuracy on labeled: %f \n', sum(Yfl==Yl)/length(Yl));
    acc = [acc, sum(Yfu==Yu)/length(Yu)];
    fprintf('Accuracy on unlabeled: %f \n', acc(end));
    
end

end