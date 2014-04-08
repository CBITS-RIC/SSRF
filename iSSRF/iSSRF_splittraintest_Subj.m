%Inductive Semi-Supervised RF
%Splittrain: split train data into labeled and unlabeled
%Split Train and test (labeled and unlabeled) between different subjects

%% Load dataset
close all, clear all; clc
Y = load('../Data/UCIHARDataset/train/y_train.txt');
X = load('../Data/UCIHARDataset/train/X_train.txt');
subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);
n_train = 3;
n_test = 15;
acc = [];

%% RF

Nsamples_vec = [100 200 400]; %n of samples per epoch
for s = 1:length(Nsamples_vec)
    
    rng('default')   %fix random number generator seed 

    
    Nsamples = Nsamples_vec(s);
    
    %randomly select subject
    % ind = randperm(length(unique(subjects)));
    % subj_train = ind(1:n_train);
    % subj_test  = ind(n_train+1:n_train+n_test);
    
    subj_train = subject_codes(1:n_train);
    subj_test  = subject_codes(n_train+1:n_train+n_test);
    
    ind_train = [];
    for i=1:n_train,
        ind_train = [ind_train; find(subjects==subj_train(i))];
    end
    ind_test = [];
    for i=1:n_test,
        ind_test = [ind_test; find(subjects==subj_test(i))];
    end
    
    %Labeled (train) and Unlabeled (test) data
    Xtrain = X(ind_train,:);   Ytrain = Y(ind_train);          %labeled data
    Xtest = X(ind_test,:);  Ytest = Y(ind_test);   %unlabeled data
    disp(sprintf('%d training datapoints', size(Xtrain,1)));
    disp(sprintf('%d test datapoints', size(Xtest,1)));
    
    %  %sequentially select data
    %     split = floor(size(X,1)*frac);
    %     Xtrain = X(1:split,:);   Ytrain = Y(1:split);          %labeled data
    %     Xtest = X(split+1:end,:);   Ytest = Y(split+1:end);            %unlabeled data
    
    
    
    %Train RF
    disp('RF Train');
    
    ntrees = 50;
    OOBVarImp = 'off';   %enable variable importance measurement
    
    forest = TreeBagger(ntrees,Xtrain,Ytrain','OOBVarImp',OOBVarImp);
    
    %one classification tree
    % tree = classregtree(features,codesTrue,'method','classification');
    % view(tree,'mode','graph') % graphic description
    
    %plot oobe
    if strcmp(OOBVarImp,'on')
        figure
        plot(oobError(forest));
        
        %plot var importance
        if strcmp(OOBVarImp,'on')
            figure('name',['k-fold ' num2str(k)])
            barh(forest.OOBPermutedVarDeltaError);
            set(gca,'ytick',1:length(trainingClassifierData.featureLabels),'yticklabel',trainingClassifierData.featureLabels)
        end
    end
    
    
    
    
    
    %% Predict and Use Unlabeled Data
    %Predict
    disp('RF Test');
    [codesRF,P_RF] = predict(forest,Xtest);
    
    %one classification tree
    % codesRF = eval(tree,features);
    
    
    %results
    codesRF = cell2vec(codesRF);
    matRF = confusionmat(Ytest,codesRF);
    accRF = length(find(codesRF==Ytest))/length(Ytest);
    disp(['accRF = ' num2str(accRF)])
    acc(s,1) = accRF;  %save accuracy
    
    
    
    %Params
    epochs = 10; % # of times unlabeled data is sampled
    RFconf = 0.8; %the confidence of RF for label prediction
    
    ind = randsample(size(Xtest,1),size(Xtest,1),false); %randomly select unlabeled data from test set
    % Nsamples = 50;                     %n of samples per epoch
    
    if epochs*Nsamples >= size(Xtest,1),
        error('too many unlabeled samples')
    end
    
    %incorporate unlabeled data
    for k = 1:epochs
        Xunl = Xtest(ind(Nsamples*(k-1)+1:Nsamples*k),:);    %new unlabeled features
        
        [codesRF,P_RF] = predict(forest,Xunl);  %predict labels for unlabeled (test) data
        ind_conf = find(max(P_RF,[],2) <= RFconf);
        
        disp(sprintf('%d of %d selected', length(ind_conf), Nsamples));
        
        codesRF = codesRF(ind_conf);  %select datapoints which exceed confidence threshold
        Xunl = Xunl(ind_conf,:);
        
        Xtrain = [Xtrain;Xunl]; %train dataset
        Ytrain = [Ytrain; cell2vec(codesRF)];
        
        %Train RF on new labeled data + previous data (can we do incremental
        %training with RF?)
        forest = TreeBagger(ntrees,Xtrain,Ytrain,'OOBVarImp',OOBVarImp);
        
        %predict on the whole test data
        [codesRF,P_RF] = predict(forest,Xtest);  %predict labels for test data
        
        accRF = length(find(cell2vec(codesRF)==Ytest))/length(Ytest);
        disp(['accRF = ' num2str(accRF)])
        acc(s,k+1) = accRF;  %save accuracy
        
        
    end


end

for s = 1:length(Nsamples_vec)
    l{s} = num2str(Nsamples_vec(s));
end

figure, hold on
plot(acc','LineWidth',2)
legend(l)
xlabel('epochs')
ylabel('accuracy')




