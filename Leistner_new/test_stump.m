clear all;
close all;

load '../Data/G50C/g50c.mat';
Y = y;
clear y;

nstumps_range = [1:5:20, 40:20:600];
% nstumps_range = 400;

ind_train = [1:round(length(X)/3), round(length(X)/2):round(5*length(X)/6)];

%our split
% Xtrain = X(ind_train, :);
% Ytrain = Y(ind_train);
% 
% Xtest = X;
% Ytest = Y;
% Xtest(ind_train, :) = [];
% Ytest(ind_train) = [];

%dataset's split
Xtrain = X(idxLabs(1,:),:);
Ytrain = Y(idxLabs(1,:));
Xtest = X(idxUnls(1,:),:);
Ytest = Y(idxUnls(1,:));

% return;
acc = [];
for nstumps = nstumps_range,
    for i=1:nstumps,
        stump{i} = ClassificationTree.fit(Xtrain, Ytrain, 'Prune','off', 'nvartosamp', 1, 'MinParent', size(Xtrain,1));
        [ys(:,i) p(:,:,i)] = predict(stump{i}, Xtest);
        %     sum(ys==Ytest)/length(Ytest)
        %     classes = stump{i}.NodeClass;
    end
    yf = mode(ys,2);
    acc = [acc, sum(yf==Ytest)/length(Ytest)];
    clear stump ys p yf;
end

plot(nstumps_range, acc, '*');
xlabel('number of stumps');
ylabel('accuracy');