%Inductive Semi-Supervised RF
%Splittrain: split train data into labeled and unlabeled
close all
ifrac = 0;
Y = load('../Data/UCIHARDataset/train/y_train.txt');
X = load('../Data/UCIHARDataset/train/X_train.txt');

frac_range = exp(-4.6052:.5:-0.1054);

for frac = frac_range,
    
    frac
    
    ifrac = ifrac+1;
    acc{ifrac} = [];
    
    clearvars -except frac ifrac X Y acc;
    
    
    
    %Randomly selects a portion of labeled data. Rest is Unlabeled
    % frac = 0.05;
    % labeled = zeros(size(X,1),1);
    % labeled(randperm(size(X,1),round(size(X,1)*frac))) = 1;
    % labeled = logical(labeled);
    % Xtrain = X(labeled,:);   Ytrain = Y(labeled);          %labeled data
    % Xtest = X(~labeled,:);   Ytest = Y(~labeled);            %unlabeled data
    
    %sequentially select data
    split = floor(size(X,1)*frac);
    Xtrain = X(1:split,:);   Ytrain = Y(1:split);          %labeled data
    Xtest = X(split+1:end,:);   Ytest = Y(split+1:end);            %unlabeled data
    
    disp(sprintf('%d training datapoints', size(Xtrain,1)));
    disp(sprintf('%d test datapoints', size(Xtest,1)));
    
    %Train RF
    disp('RF Train');
    
    ntrees = 10;
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
    
    
    
    %% Test RF
    disp('RF Test');
    
    
    % Ytest = load('../Data/UCIHARDataset/test/y_test.txt');
    
    % Xtest = load('../Data/UCIHARDataset/test/X_test.txt');
    
    %% Predict and Use Unlabeled Data
    %Predict
    [codesRF,P_RF] = predict(forest,Xtest);
    
    %one classification tree
    % codesRF = eval(tree,features);
    
    
    %results
    codesRF = cell2vec(codesRF);
    matRF = confusionmat(Ytest,codesRF);
    accRF = length(find(codesRF==Ytest))/length(Ytest);
    disp(['accRF = ' num2str(accRF)])
    acc{ifrac} = [acc{ifrac};accRF];  %save accuracy
    
    % figure
    % imagesc(matRF), colormap gray
    % figure, hold on
    % plot(codesTrue,'g'); plot(codesRF,'r')
    
    %Params
    % p = 0.005;   % perc of test data to use as unlabeled
    epochs = 10; % # of times unlabeled data is sampled
    RFconf = 0; %the confidence of RF for label prediction
    
    ind = randsample(size(Xtest,1),size(Xtest,1),false); %randomly select unlabeled data from test set
    Nsamples = 5;%floor(p*size(Xtest,1));                      %n of samples per epoch
    
    if epochs*Nsamples >= size(Xtest,1),
        error('too many unlabeled samples')
    end
    
    for k = 1:epochs
        Xunl = Xtest(ind(Nsamples*(k-1)+1:Nsamples*k),:);    %new unlabeled features
        
        [codesRF,P_RF] = predict(forest,Xunl);  %predict labels for unlabeled (test) data
        ind_conf = find(max(P_RF,[],2) > RFconf);
        
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
        acc{ifrac} = [acc{ifrac};accRF];  %save accuracy
        
        
    end
    
end









