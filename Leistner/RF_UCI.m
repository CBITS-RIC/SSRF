%%Train and Test RF on UCI dataset

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

codesTrue = load('../Data/UCIHARDataset/train/y_train.txt');

features = load('../Data/UCIHARDataset/train/X_train.txt');

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

ntrees = 10;
OOBVarImp = 'off';   %enable variable importance measurement

forest = TreeBagger(ntrees,features,codesTrue','OOBVarImp',OOBVarImp);

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

%Load Test data
% accx = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_x_test.txt');  
% accy = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_y_test.txt');  
% accz = load ('../Data/UCIHARDataset/test/InertialSignals/total_acc_z_test.txt');  

codesTrue = load('../Data/UCIHARDataset/test/y_test.txt');

features = load('../Data/UCIHARDataset/test/X_test.txt');

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

%Scale features ([0,1]
% featureMax = max(features,[],1);
% featureMin = min(features,[],1);
% features = (features - repmat(featureMin,size(features,1),1))*spdiags(1./(featureMax-featureMin)',0,size(features,2),size(features,2));

%Predict 
[codesRF,P_RF] = predict(forest,features);

%one classification tree
% codesRF = eval(tree,features);


%results
codesRF = cell2vec(codesRF);
matRF = confusionmat(codesTrue,codesRF);
accRF = length(find(codesRF==codesTrue))/length(codesTrue);
disp(['accRF = ' num2str(accRF)])
figure
imagesc(matRF), colormap gray
figure, hold on
plot(codesTrue,'g'); plot(codesRF,'r')




