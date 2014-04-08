%%SSRF with DA from Leistner, Saffari et al. 2009 paper
%Authors: Luca Lonini and Sohrab Saeb
%Use modified Matlab TreeBagger class (X,Y properties set to public)
%Added random selection of indices for labeled/unlabeled (idxLabs,idxUnls)
%save history of optimal probability Pu_opt (not done yet!)
%Multiclass version: change definition of margin loss for more than 2 classes
%Fix: Prevents Nan if Pu_opt is 0 

clear all;
close all
tic;
rng('default')   %fix random number generator seed 
%parameters
ntrees = 50;     %forest size
T0 = 5;          %initial temperature
alpha = 1;     %coeff to control the weight of the unlabeled part in the loss function
epochs = 200;     %epochs for unlabeled training
tau = 50;  %cooling fcn time constant



%% Load  Fisher iris dataset

% frac = 0.1;    %the frac of labeled data

% X = meas;   %predictors
% Y = zeros(size(species));   %numeric labels
% %assign numeric labels
% labels = unique(species);
% for i = 1:length(labels)
%     l = labels(i); 
%     ind = strcmp(species,l);
%     Y(ind) = i;
% end
  

% n_class = length(unique(Y)); %the # of classes
% classes = unique(Y);         %the class codes
% 
% %Randomly selects a portion of labeled data. Rest is Unlabeled
% labeled = zeros(size(X,1),1);
% labeled(randperm(size(X,1),round(size(X,1)*frac))) = 1;
% labeled = logical(labeled);
% Xl = X(labeled,:);   Yl = Y(labeled);          %labeled data
% Xu = X(~labeled,:);  Yu = Y(~labeled);   %unlabeled data
% Yu_forest = zeros(ntrees*size(Xu,1),1);          %unlabeled data for entire forest


%% Load UCI Dataset

frac = 0.1;    %the frac of labeled data


disp('Load UCI Dataset')

Xtrain = load('../Data/UCIHARDataset/train/X_train.txt');
Ytrain = load('../Data/UCIHARDataset/train/y_train.txt');
% Xtest = load('../Data/UCIHARDataset/test/X_test.txt');   
% Ytest = load('../Data/UCIHARDataset/test/y_test.txt');   

subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);
n_train = 1;
n_test = 1;

%%this should be made such that the training and test subjects are picked
%randomly
subj_train = subject_codes(1:n_train);
subj_test  = subject_codes(n_train+1:n_train+n_test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind_train = [];
for i=1:n_train,
    ind_train = [ind_train; find(subjects==subj_train(i))];
end
ind_test = [];
for i=1:n_test,
    ind_test = [ind_test; find(subjects==subj_test(i))];
end

n_class = length(unique(Ytrain)); %the # of classes
classes = unique(Ytrain);         %the class codes

%Randomly selects a portion of training data as labeled. Rest is Unlabeled
% labeled = zeros(size(Xtrain,1),1);
% labeled(randperm(size(Xtrain,1),round(size(Xtrain,1)*frac))) = 1;
% labeled = logical(labeled);
% Xl = Xtrain(labeled,:);   Yl = Ytrain(labeled);          %labeled data
% Xu = Xtrain(~labeled,:);  Yu = Ytrain(~labeled);   %unlabeled data
Xl = Xtrain(ind_train,:);   Yl = Ytrain(ind_train);          %labeled data
Xu = Xtrain(ind_test,:);  Yu = Ytrain(ind_test);   %unlabeled data
Yu_forest = zeros(ntrees*size(Xu,1),1);          %unlabeled data for entire forest


%% Train RF on labeled data
disp('Train RF on labeled data')

forest = TreeBagger(ntrees,Xl,Yl,'OOBPred','on');
oobind = forest.OOBIndices;
%compute out-of-bag error for each tree
GE = zeros(ntrees,1);
Pl_forest = zeros(length(Yl),n_class);

% Computing the forest accuracy on unlabeled data
[Yfu,Pu_forest] = predict(forest,Xu);
Yfu = str2num(cell2mat(Yfu));
acc(1) = sum(Yu==Yfu)/length(Yu)


%% DA optimization

lgs=[];
for m = 1:epochs,
    T = T0*exp(-(m-1)/tau);     %reduce temp value
    Tvals(m) = T;               %save T values
    
    [Yu_p,Pu] = predict(forest,Xu);    %compute prob Pu_i of each class (i) for the unlabeled data (output prob of forest)
    
    %     Yu_p = str2num(cell2mat(Yu_p));
    %     [r0,c0] = find(Pu == 0);      %find zero P values (to prevent log(0)=-inf)
    %     Pu(r0,c0) = eps;
    %     Pu = Pu - .5;
    
    %margin max loss fcn (Entropy)
    for c = 1:n_class
        lg(:,c) = exp(-2*(Pu(:,c)-max(Pu(:,classes(classes ~= c)),[],2)));
        lgs = [lgs; mean(mean(lg))];
    end
    
    %Compute Optimal Distribution over predicted labels
    Pu_opt = exp(-(alpha*lg+T)/T);
    Pu_opt = max(Pu_opt,eps);       %prevents Nan if Pu_opt is 0
    Z = sum(Pu_opt,2); Z = repmat(Z,[1 n_class]);
    Pu_opt = Pu_opt./Z; %normalized probabilities
    
    
    %draw random label from Pu_opt distribution for each unlabeled data
    %point and each tree
    for p = 1:length(Yu)
        Yu_temp = randsample(1:n_class,ntrees,true,Pu_opt(p,:));  %predicted label for point p
        Yu_forest(p:length(Yu):length(Yu)*(ntrees-1)+p,1) = Yu_temp;
    end
    
     
    Xl_forest = repmat(Xl,[ntrees,1]); Xu_forest = repmat(Xu,[ntrees 1]);
    Yl_forest = repmat(Yl,[ntrees 1]); 
    X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
    
    %train forest on labeled and unlabeled data
    %Each tree is grown 
    forest = TreeBagger(ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/ntrees);
%     oob_tmp = oobError(forest);    
%     oob_total(m) = oob_tmp(end);    
    oobind = forest.OOBIndices;
    
    %find indices of out-of-bag labeled data
    OOBM = ones(length(Yl),ntrees);
    
    %produce oobindex for labeled data 
    for i = 1:ntrees
        OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
        OOBM = OOBM.*OOBM_tmp;
    end
    
    %compute out-of-bag error for each tree
%     GE = [];
%     for t = 1:ntrees
%         if sum(OOBM(:,t))==0,
%             continue;
%         end
%         [Yp,Pl] = predict(forest,Xl(logical(OOBM(:,t)),:),'trees', t);   %Pl is the prob of each tree
%         Yp = str2num(cell2mat(Yp));
%         acc = mean((Yp-Yl(logical(OOBM(:,t))))==0);
%         GE = [GE, 1-acc];  %generalization error for 1 tree
%     end
  
    [Yfu,Pu_forest] = predict(forest,Xu);
    Yfu = str2num(cell2mat(Yfu));
    acc(m+1) = sum(Yu==Yfu)/length(Yu)

%     GEforest(m+1) = mean(GE);
    
        
    %compute predictions (scores or prob) for each out-of-bag datapoint
%     oobe = zeros(length(Yl),1);
%     for p = 1:length(Yl)
%         trees = find(OOBM(p,:) == 1);   %trees for which point p is out-of-bag
%         [Yp,Pl] = predict(forest,Xl(p,:),'trees', trees);   %Pl is the prob of each tree 
%         ind=find(ismember(classes,Yl(end)));                %the true class code
%         Ptrue = zeros(1,length(classes)); Ptrue(ind) = 1;   %the true prob vector for the class
%         oobe(p) = norm(Pl-Ptrue);
%     end
%     
%     oobetot(m) = mean(oobe);
end


%% Results
figure
% plot(GEforest)
subplot(211), plot(acc,'LineWidth',2);
subplot(212), plot(Tvals,'LineWidth',2)

figure
bar([acc(1) acc(end)])

% figure(2)
% plot(lgs);

% figure(3);
% plot(oob_total);

toc;