clc
close all
clear all
epochs = 50;
% tau = [10 20 30 40];
T0 = 1;%1.25;
% alpha = [0.75 1.25 1.5]; 
subj = 3;   %total # of train (labeled) subjects to use
% F = cell(length(tau),1);  %cell array containing all the trained forests
acc = zeros(length(T0),epochs+1,subj); %matrix with accuracy over unlabeled data for every subject
oobe = acc;                            %matrix with oobe for every subject and epoch
acc_l = acc;                           %matrix with accuracy over labeled data for every subject
Nte = 3;    %# of subject to use as test (unlabeled)

%main loop over # of train vs test subjects
for Ntr = 1%3
    
    fprintf('# of Training subjects: %d \n', Ntr);

    %loop over model parameters (tau, alpha, etc)
    for k = 1:10%:length(tau)
        k
%         F = configUCI_fewsamples(Ntr, Nte); %initialize a forest - specify Ntr and Nte subjects
        F = configUCI_fewsamples(2, 0);
%         F = configUCI_crosstrial(1);
        
        F.tau = 40;%tau(k);          %set tau (80)
        %     F{k}.alpha = alpha(k);    %set alpha
        F.T0 = T0;%(k);           %set T0
        
        tic 
        F.trainforest_multic(epochs, true);    %train ssrf

        toc
        
        acc(k,:,Ntr) = F.acc;   %save accuracy over parameters
        acc_l(k,:,Ntr) = F.acc_l;   %accuracy for labeled data
        oobe(k,:,Ntr) = F.oobe; %save OOBE
        Pl_forest{k} = F.Pl;    %probability over labeled data
        Pu_forest{k} = F.Pu;    %probability over labeled data
        confmat{k} = F.confmat; %confusion matrix over unlabeled data at the final iteration

        clear F
        
    end
    
end

toc

% save 'accforests.mat' acc;

%%Plot improvement in accuracy over many sims (k)
I = [acc(:,1,1) acc(:,end,1)];
figure('name','2 samples'), hold on
bar(I), ylim([0.5 1])
figure
I = acc(:,end,1)-acc(:,1,1);
bar(I);
mean(I)

%compute Entropy over unlabeled and labeled data
% close all

E = [];
for i=1:k,
    Pl_forest{i}(Pl_forest{i} == 0) = eps
    E_ = -mean(sum(Pl_forest{i}.*log2(Pl_forest{i}),2));
    E = [E E_(:)];
end
figure, hold on
plot(E(:,I>=0),'b','LineWidth',2)
plot(E(:,I<0),'r','LineWidth',2)

E = [];
for i=1:k,
    Pu_forest{i}(Pu_forest{i} == 0) = eps
    E_all = -sum(Pu_forest{i}.*log2(Pu_forest{i}),2);
%     E_all = E_all(:,:);
    E_ = E_all;
    E_(E_<=.7) = NaN;
    E_ = nanmean(E_);
    E = [E E_(:)];
end
figure, hold on
plot(E(:,I>=0),'b','LineWidth',2)
plot(E(:,I<0),'r','LineWidth',2)


% save('trainedforests_40trees_1tr_6te_tau70.mat','F')

% figure, hold on
% plot(acc(:,:,3)','LineWidth',2)
% for s = 1:length(tau)
%     l{s} = num2str(tau(s));
% end
% 
% legend(l)