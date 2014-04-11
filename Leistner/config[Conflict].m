function forest = config()

%parameters
ntrees = 10;     %forest size
T0 = 5;          %initial temperature
alpha = 1;     %coeff to control the weight of the unlabeled part in the loss function
tau = 60;  %cooling fcn time constant


%% g50c dataset
% load g50c.mat;
% Y = y/2+1.5;
% n_class = length(unique(Y)); %the classes
% Xl = X(idxLabs(1,:),:);    Yl = Y(idxLabs(1,:));
% Xu = X(idxUnls(1,:),:);    Yu = Y(idxUnls(1,:));

%% UCI Dataset

frac = 0.1;    %the frac of labeled data


disp('Load UCI Dataset')

Xtrain = load('../Data/UCIHARDataset/train/X_train.txt');
Ytrain = load('../Data/UCIHARDataset/train/y_train.txt');
% Xtest = load('../Data/UCIHARDataset/test/X_test.txt');   
% Ytest = load('../Data/UCIHARDataset/test/y_test.txt');   

%Randomly selects a portion of training data as labeled. Rest is Unlabeled
% labeled = zeros(size(Xtrain,1),1);
% labeled(randperm(size(Xtrain,1),round(size(Xtrain,1)*frac))) = 1;
% labeled = logical(labeled);
% Xl = Xtrain(labeled,:);   Yl = Ytrain(labeled);          %labeled data
% Xu = Xtrain(~labeled,:);  Yu = Ytrain(~labeled);   %unlabeled data

%Select # of subjects for labeled (train) and unlabeled (test) 
subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);
n_train = 3;
n_test = 3;

%randomly select subject
% rng('default')   %fix random number generator seed
% 
% ind = randperm(length(unique(subjects)));
% subj_train = ind(1:n_train);
% subj_test  = ind(n_train+1:n_train+n_test);

%ordered selection of subjects
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

n_class = length(unique(Ytrain)); %the # of classes
% classes = unique(Ytrain);         %the class codes


Xl = Xtrain(ind_train,:);   Yl = Ytrain(ind_train);          %labeled data
Xu = Xtrain(ind_test,:);  Yu = Ytrain(ind_test);   %unlabeled data
% Yu_forest = zeros(ntrees*size(Xu,1),1);          %unlabeled data for entire forest

%% Params to init forests

PARAM = {ntrees,T0,alpha,tau,Xl,Yl,Xu,Yu,n_class};

forest = ssforest(PARAM);

end