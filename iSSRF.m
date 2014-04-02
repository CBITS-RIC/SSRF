%Inductive Semi-Supervised RF

clear all
close all

%%LOAD Train Data
% accx = load ('../Data/UCIHARDataset/train/InertialSignals/total_acc_x_train.txt');
% accy = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_y_train.txt');
% accz = load('../Data/UCIHARDataset/train/InertialSignals/total_acc_z_train.txt');
%
% gyrx = load ('../Data/UCIHARDataset/train/InertialSignals/body_gyro_x_train.txt');
% gyry = load('../Data/UCIHARDataset/train/InertialSignals/body_gyro_y_train.txt');
% gyrz = load('../Data/UCIHARDataset/train/InertialSignals/body_gyro_z_train.txt');

Ytrain = load('../Data/UCIHARDataset/train/y_train.txt');

Xtrain = load('../Data/UCIHARDataset/train/X_train.txt');

%Store Code and label of each unique State
% StateCodes = cell(length(uniqStates),2);
% StateCodes(:,1) =
% StateCodes(:,2) = unique(labels); %sorted by unique


%extract features
% features_acc = []; features_gyr = []; features = [];
% disp('extracting features')
% for k = 1:size(accx,1)
%
%     acc = [accx(k,:);accy(k,:);accz(k,:)];
%     [fvec, flab_acc] = getFeatures(acc);
%     features_acc = [features_acc;fvec;];
%
%     gyr = [gyrx(k,:);gyry(k,:);gyrz(k,:)];
%     [fvec, flab_gyr] = getFeatures(gyr);
%     features_gyr = [features_gyr;fvec;];
%
%     features = [features_acc features_gyr];
%     flab = [flab_acc;flab_gyr];
% end
% disp('Features extracted')
%
% %Scale features ([0,1]
% featureMax = max(features,[],1);
% featureMin = min(features,[],1);
% features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));


%Train RF
disp('RF Train');

ntrees = 100;
mtry = round(sqrt(size(Xtrain,2)));
OOBVarImp = 'off';   %enable variable importance measurement

%MATLAB Implementation
% forest = TreeBagger(ntrees,Xtrain,Ytrain','OOBVarImp',OOBVarImp);


%one classification tree
% tree = classregtree(features,codesTrue,'method','classification');
% view(tree,'mode','graph') % graphic description

%plot oobe
% if strcmp(OOBVarImp,'on')
%     figure
%     plot(oobError(forest));
%     
%     %plot var importance
%     if strcmp(OOBVarImp,'on')
%         figure('name',['k-fold ' num2str(k)])
%         barh(forest.OOBPermutedVarDeltaError);
%         set(gca,'ytick',1:length(trainingClassifierData.featureLabels),'yticklabel',trainingClassifierData.featureLabels)
%     end
% end


%R implementation of random forest(uses Mex file)
forest = classRF_train(Xtrain,Ytrain',ntrees,mtry); %the train set does not contain <80%threshold clips




%% Test RF
disp('RF Test');

%Load Test data
% accx = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_x_test.txt');
% accy = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_y_test.txt');
% accz = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_z_test.txt');

Ytest = load('../Data/UCIHARDataset/test/y_test.txt');

%extract features
% features_acc = []; features_gyr = []; features = [];
% disp('extracting features')
% for k = 1:size(accx,1)
%
%      acc = [accx(k,:);accy(k,:);accz(k,:)];
%     [fvec, flab_acc] = getFeatures(acc);
%     features_acc = [features_acc;fvec;];
%
%     gyr = [gyrx(k,:);gyry(k,:);gyrz(k,:)];
%     [fvec, flab_gyr] = getFeatures(gyr);
%     features_gyr = [features_gyr;fvec;];
%
%     features = [features_acc features_gyr];
% end
% disp('Features extracted')
Xtest = load('../Data/UCIHARDataset/test/X_test.txt');

%Scale features ([0,1]
% featureMax = max(features,[],1);
% featureMin = min(features,[],1);
% features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));

%% Predict and Use Unlabeled Data
acc = [];
%Predict
% [codesRF,P_RF] = predict(forest,Xtest);

%one classification tree
% codesRF = eval(tree,features);

%R Implementation
test_options.predict_all = 1; %returns prediction per tree and votes
[codesRF, votes, prediction_per_tree] = classRF_predict(Xtest,forest,test_options); 
P_RF = votes./forest.ntree;  %Prob of each class



%results
codesRF = cell2vec(codesRF);
matRF = confusionmat(Ytest,codesRF);
accRF = length(find(codesRF==Ytest))/length(Ytest);
disp(['accRF = ' num2str(accRF)])
acc = [acc;accRF];  %save accuracy

% figure
% imagesc(matRF), colormap gray
% figure, hold on
% plot(codesTrue,'g'); plot(codesRF,'r')

%Params
p = .1;   % perc of test data to use as unlabeled
epochs = 10; % # of times unlabeled data is sampled
RFconf = 1.1; %the confidence of RF for label prediction

ind = randsample(size(Xtest,1),size(Xtest,1),false); %randomly select unlabeled data from test set
Nsamples = floor(p*size(Xtest,1));                      %n of samples per epoch

if epochs*p > 1
    error('too many unlabeled samples')
end

for k = 1:epochs
    Xunl = Xtest(ind(Nsamples*(k-1)+1:Nsamples*k),:);    %new unlabeled features
    
    %Predict - MATLAB
%     [codesRF,P_RF] = predict(forest,Xunl);  %predict labels for unlabeled (test) data
    
    %Predict - R
    [codesRF, votes, prediction_per_tree] = classRF_predict(Xunl,forest,test_options); 
	P_RF = votes./forest.ntree;  %Prob of each class

    ind_conf = find(max(P_RF,[],2) < RFconf); 
    
    disp(sprintf('Samples selected = %d of %d',length(ind_conf),Nsamples));
    
    codesRF = codesRF(ind_conf);  %select datapoints which exceed confidence threshold
    Xunl = Xunl(ind_conf,:);
    
    Xtrain = [Xtrain;Xunl]; %train dataset
    Ytrain = [Ytrain; cell2vec(codesRF)];
    
    %Train RF on new labeled data + previous data (can we do incremental
    %training with RF?)
    clear forest; rng('shuffle');
    
    %Train - MATLAB
%     forest = TreeBagger(ntrees,Xtrain,Ytrain,'OOBVarImp',OOBVarImp);
    
    %Train - R
    forest = classRF_train(Xtrain,Ytrain',ntrees,mtry); %the train set does not contain <80%threshold clips

    %predict on the whole test data - Matlab
%     [codesRF,P_RF] = predict(forest,Xtest);  %predict labels for test data
    
    %predict - R
    [codesRF, votes, prediction_per_tree] = classRF_predict(Xtest,forest,test_options);

    
    accRF = length(find(cell2vec(codesRF)==Ytest))/length(Ytest);
    disp(['accRF = ' num2str(accRF)])
    acc = [acc;accRF];  %save accuracy
    
    
end







