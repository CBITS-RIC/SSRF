%% class for SSRF

classdef ssforest < handle
    properties
        
        ntrees;
        T0;        %initial temperature
        alpha;     %coeff to control the weight of the unlabeled part in the loss function
        tau;       %cooling fcn time constant
        Xl;
        Yl;
        Xu;
        Yu;
        n_class;
        acc; 
        acc_l;  %accuracy on labeled data
        Pl;  %forest probability over labeled data
        Pu;  %forest probability over unlabeled data
        oobe;   %out of bag (generalization) error
        confmat;    %confusion matrix at the end for unlabeled data
        Tvals;
    end
    
    methods
        function obj = ssforest(PARAM)
            
            obj.ntrees = PARAM{1};
            obj.T0 = PARAM{2};
            obj.alpha = PARAM{3};
            obj.tau = PARAM{4};
            obj.acc = [];
            obj.acc_l = [];
            obj.Pl = {};
            obj.Pu = [];
            obj.oobe = [];
            obj.confmat = [];
            obj.Tvals = [];
            
            
            %dataset
            obj.Xl = PARAM{5}; obj.Yl = PARAM{6};
            obj.Xu = PARAM{7}; obj.Yu = PARAM{8};
            obj.n_class = PARAM{9};
            
        end
        
        
        
        function [acc, Tvals] = trainforest_multic(this, epochs, repeat)
            
%             rng('default')   %fix random number generator seed
%             rng('shuffle')   %reset random number generator seed


            %labeled and unlabeled data
            Xl = this.Xl; Yl = this.Yl;
            Xu = this.Xu; Yu = this.Yu;
            Yu_forest = zeros(this.ntrees*size(Xu,1),1);          %unlabeled data for entire forest
            classes = 1:this.n_class;                             %the class codes
            
            %% Train RF on labeled data
            
            fprintf('Train RF (%d trees) on labeled data...\n', this.ntrees);
            Xl_orig  = Xl;
            Yl_orig = Yl;

            if repeat,
                RepFac = ceil(eps+this.T0*length(Yu)/length(Yl_orig));
                Xl = repmat(Xl_orig, RepFac, 1);
                Yl = repmat(Yl_orig, RepFac, 1);
                fprintf('\n %d initial labeled samples', size(Xl,1));
            end

            forest = TreeBagger(this.ntrees,Xl,Yl,'OOBPred','on');
            
            % Computing the forest accuracy on unlabeled data
            [Yfu,Pu_forest] = predict(forest,Xu);
            Yfu = str2num(cell2mat(Yfu));
            acc(1) = sum(Yu==Yfu)/length(Yu);
            fprintf('\n %f', acc(1));
            this.Pu(:,:,1) = Pu_forest;        %forest probability on unlabeled data
                     
            %compute accuracy on labeled data 
            [Yfl,Pl_forest] = predict(forest,Xl);
            Yfl = str2num(cell2mat(Yfl));
            this.acc_l(1) = mean(Yl==Yfl);
            this.Pl{1} = Pl_forest;        %forest probability on labeled data
            
            %% compute predictions (scores or prob) for each out-of-bag datapoint
            %find indices of out-of-bag labeled data
            oobind = forest.OOBIndices;
            GE = zeros(this.ntrees,1);
            for t = 1:this.ntrees
                %check if the tree is observing all the samlples
                if ~sum(oobind(:,t)),
%                     disp('oobind: Found a tree which observes all the samples.');
                    GE(t) = nan;
                    continue;
                end
                [Yp,~] = predict(forest,Xl(logical(oobind(:,t)),:),'trees', t);   %Pl is the prob of each tree
                Yp = str2num(cell2mat(Yp));
                acc_oobe = mean(Yp==Yl(logical(oobind(:,t))));
                GE(t) = 1-acc_oobe;  %generalization error for 1 tree
            end
            this.oobe(1) = nanmean(GE);
            
            %% DA optimization
            
            lgs=[];
            for m = 1:epochs,
                T = this.T0*exp(-(m-1)/this.tau);     %reduce temp value
                Tvals(m) = T;               %save T values
                
                [Yu_p,Pu] = predict(forest,Xu);    %compute prob Pu_i of each class (i) for the unlabeled data (output prob of forest)
                
                %margin max loss fcn (Entropy)
                for c = 1:this.n_class
                    lg(:,c) = exp(-2*(Pu(:,c)-max(Pu(:,classes(classes ~= c)),[],2)));
                    lgs = [lgs; mean(mean(lg))];
                end
                
                %Compute Optimal Distribution over predicted labels
                Pu_opt = exp(-(this.alpha*lg+T)/T);
                Pu_opt = max(Pu_opt,eps);       %prevents Nan if Pu_opt is 0
                Z = sum(Pu_opt,2); Z = repmat(Z,[1 this.n_class]);
                Pu_opt = Pu_opt./Z; %normalized probabilities
                
                
                %draw random label from Pu_opt distribution for each unlabeled data
                %point and each tree
                for p = 1:length(Yu)
                    Yu_temp = randsample(1:this.n_class,this.ntrees,true,Pu_opt(p,:));  %predicted label for point p
                    Yu_forest(p:length(Yu):length(Yu)*(this.ntrees-1)+p,1) = Yu_temp;
                end
                
                
                if repeat,
                    RepFac = ceil(eps+T*length(Yu)/length(Yl_orig));
                    Xl = repmat(Xl_orig, RepFac, 1);
                    Yl = repmat(Yl_orig, RepFac, 1);
                    fprintf('\n %d labeled samples', size(Xl,1));
                end
                
                Xl_forest = repmat(Xl,[this.ntrees,1]); Xu_forest = repmat(Xu,[this.ntrees 1]);
                Yl_forest = repmat(Yl,[this.ntrees 1]);
                X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
                
                %train forest on labeled and unlabeled data
                %Each tree is grown
                forest = TreeBagger(this.ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/this.ntrees);
                %     oob_tmp = oobError(forest);
                %     oob_total(m) = oob_tmp(end);
                
                
                %compute accuracy on unlabeled data
                [Yfu,Pu_forest] = predict(forest,Xu);
                Yfu = str2num(cell2mat(Yfu));
                acc(m+1) = sum(Yu==Yfu)/length(Yu);
                fprintf('\n %f', acc(m+1));    %print accuracy over training
                this.Pu(:,:,m+1) = Pu_forest;        %forest probability on unlabeled data

                %compute accuracy on labeled data
                [Yfl,Pl_forest] = predict(forest,Xl);
                Yfl = str2num(cell2mat(Yfl));
                this.acc_l(m+1) = mean(Yl==Yfl);
                this.Pl{m+1} = Pl_forest;        %forest probability on labeled data

                %% compute predictions (scores or prob) for each out-of-bag datapoint
                %find indices of out-of-bag labeled data
                oobind = forest.OOBIndices;
                OOBM = ones(length(Yl),this.ntrees);
                %produce oobindex for labeled data
                for i = 1:this.ntrees
                    OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
                    OOBM = OOBM.*OOBM_tmp;
                end
                GE = zeros(this.ntrees,1);
                for t = 1:this.ntrees
                    %check if the tree is observing all the training
                    %samples
                    if ~sum(OOBM(:,t)),
%                         disp('OOBM: Found a tree which observes all the samples.');
                        GE(t) = nan;
                        continue;
                    end
                    [Yp,~] = predict(forest,Xl(logical(OOBM(:,t)),:),'trees', t);   %Pl is the prob of each tree
                    Yp = str2num(cell2mat(Yp));
                    acc_oobe = mean(Yp==Yl(logical(OOBM(:,t))));
                    GE(t) = 1-acc_oobe;  %generalization error for 1 tree
                end
                this.oobe(m+1) = nanmean(GE);
                
            end
            fprintf('\n');
            
            %% Results
%             figure
%             subplot(211), plot(acc,'LineWidth',2);
%             subplot(212), plot(Tvals,'LineWidth',2)
            this.confmat = confusionmat(Yu,Yfu);
            this.acc = acc;
            this.Tvals = Tvals;

        end
        
%% ***********Use also HMM to smooth Data**********************************        
         function [acc, Tvals] = trainforest_multicHMM(this,epochs)
            
%              rng('default')   %fix random number generator seed
             
             %labeled and unlabeled data
             
             Xl = this.Xl; Yl = this.Yl;
             Xu = this.Xu; Yu = this.Yu;
             Yu_forest = zeros(this.ntrees*size(Xu,1),1);          %unlabeled data for entire forest
             classes = 1:this.n_class;                             %the class codes
          
            uniqStates = unique(Yl);
            
            %% Train RF on labeled data and init HMM
            disp('Train RF on labeled data')
            
            forest = TreeBagger(this.ntrees,Xl,Yl,'OOBPred','on');
            oobind = forest.OOBIndices;
            %compute out-of-bag error for each tree
            GE = zeros(this.ntrees,1);
            Pl_forest = zeros(length(Yl),this.n_class);
            
            
             % Initialize HMM
             %compute emission prob matrix from P_RF on train data
             [~,P_RF] = predict(forest,Xl);
                          
             %inialize parameters for hmm
             d       = length(uniqStates);   %number of symbols (=#states)
             nstates = d;                    %number of states
             mu      = zeros(d,nstates);     %mean of emission distribution
             sigma   = zeros(d,1,nstates);   %std dev of emission distribution
             Pi      = ones(length(uniqStates),1) ./ length(uniqStates); %uniform prior
             sigmaC  = .1;                   %use a constant std dev
             PTrain = P_RF;   %Emission prob = RF class prob for training data
             pnt = 0.9;             %prob of remaining in current state
             A = eye(d)*pnt;        %the transition matrix
             A(A == 0) = (1-pnt)./(d-1);
             
             
             %create emission probabilities for HMM
             PBins  = cell(d,1);
             %for each type of state we need a distribution
             for bin = 1:d
                 clipInd         = find(Yl==uniqStates(bin)); %find clip indices
                 PBins{bin,1}    = PTrain(clipInd,:);    %The emission probability of the HMM is the RF prob over training data
                 mu(:,bin)       = mean(PBins{bin,1}); %mean
                 sigma(:,:,bin)  = sigmaC; %set std dev
             end
             
             %create distribution for pmtk3 package
             emission        = struct('Sigma',[],'mu',[],'d',[]);
             emission.Sigma  = sigma;
             emission.mu     = mu;
             emission.d      = d;
             
             %construct HMM using pmtk3 package
             HMMmodel           = hmmCreate('gauss',Pi,A,emission);
             HMMmodel.emission  = condGaussCpdCreate(emission.mu,emission.Sigma);
             HMMmodel.fitType   = 'gauss';
            
            % Computing the forest accuracy on unlabeled data
            [Yfu,Pu_forest] = predict(forest,Xu);
            Yfu = str2num(cell2mat(Yfu));
            %acc(1) = sum(Yu==Yfu)/length(Yu);
            
            %improve prediction by smoothing with HMM
            %predict w HMM
            [gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,Pu_forest');
            [~, codesHMM] = max(gamma);
            acc(1) = sum(Yu==codesHMM')/length(Yu);    
            
            %% DA optimization
            
            lgs=[];
            for m = 1:epochs,
                T = this.T0*exp(-(m-1)/this.tau);     %reduce temp value
                Tvals(m) = T;               %save T values
                
                [Yu_p,Pu] = predict(forest,Xu);    %compute prob Pu_i of each class (i) for the unlabeled data (output prob of forest)           
                [Pu, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,Pu');    %smooth RF prob with HMM
                Pu = Pu';
                
                %margin max loss fcn (Entropy)
                for c = 1:this.n_class
                    lg(:,c) = exp(-2*(Pu(:,c)-max(Pu(:,classes(classes ~= c)),[],2)));
                    lgs = [lgs; mean(mean(lg))];
                end
                
                %Compute Optimal Distribution over predicted labels
                Pu_opt = exp(-(this.alpha*lg+T)/T);
                Pu_opt = max(Pu_opt,eps);       %prevents Nan if Pu_opt is 0
                Z = sum(Pu_opt,2); Z = repmat(Z,[1 this.n_class]);
                Pu_opt = Pu_opt./Z; %normalized probabilities
                
                
                %draw random label from Pu_opt distribution for each unlabeled data
                %point and each tree
                for p = 1:length(Yu)
                    Yu_temp = randsample(1:this.n_class,this.ntrees,true,Pu_opt(p,:));  %predicted label for point p
                    Yu_forest(p:length(Yu):length(Yu)*(this.ntrees-1)+p,1) = Yu_temp;
                end
                
                
                Xl_forest = repmat(Xl,[this.ntrees,1]); Xu_forest = repmat(Xu,[this.ntrees 1]);
                Yl_forest = repmat(Yl,[this.ntrees 1]);
                X_forest = [Xl_forest;Xu_forest]; Y_forest = [Yl_forest;Yu_forest];
                
                %train forest on labeled and unlabeled data
                %Each tree is grown
                forest = TreeBagger(this.ntrees,X_forest,Y_forest,'OOBPred','On','FBoot',1/this.ntrees);
                %     oob_tmp = oobError(forest);
                %     oob_total(m) = oob_tmp(end);
                oobind = forest.OOBIndices;
                
                %find indices of out-of-bag labeled data
                OOBM = ones(length(Yl),this.ntrees);
                
                %produce oobindex for labeled data
                for i = 1:this.ntrees
                    OOBM_tmp = oobind((i-1)*length(Yl)+1:i*length(Yl),:);
                    OOBM = OOBM.*OOBM_tmp;
                end
                 
                % Computing the forest accuracy on unlabeled data
                [Yfu,Pu_forest] = predict(forest,Xu);
                Yfu = str2num(cell2mat(Yfu));
                %acc(m+1,1) = sum(Yu==Yfu)/length(Yu);

                %improve prediction by smoothing with HMM
                [gamma, ~, ~, ~, ~]   = hmmInferNodes(HMMmodel,Pu_forest');    %smooth RF prob with HMM
                [~, codesHMM] = max(gamma);
                acc(m+1) = sum(Yu==codesHMM')/length(Yu);
                
                acc(m+1)    %print accuracy over training
                
            end
            
            %% Results
%             figure
%             subplot(211), plot(acc,'LineWidth',2);
%             subplot(212), plot(Tvals,'LineWidth',2)
       
            this.acc = acc;
            this.Tvals = Tvals;

        end
        
        
    end
    
    
end
    
    
    
    
    
    
