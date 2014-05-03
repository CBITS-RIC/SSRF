clear all;
close all;

simtypes = {'few', 'crosssubj'};
simtype = simtypes{1};

subjects = 1;
nsamp = 1;  %for few-samples (no. samples per trial)
RepFac = 1; %for few-samples (1 = equal; 0 = no balance)
trial = 1;  %for cross-trial (trial no. for labeled)

ntrees = 400;
epochs = 20;
T0 = 1;
tau = 40;
alpha = 1;

load fisheriris;
X = meas;
Y = zeros(length(species),1);
sp = unique(species);
for i=1:length(sp),
    Y(ismember(species, sp{i})) = i;
end

for i = 1:1,%size(subjects,1),
    acc_{i} = [];
    for k = 1:10,
        tic;
        fprintf('\n******* Iteration %d:\n', k);
        switch simtype,
            case 'few',
                [Xl Yl Xu Yu] = config_fewsamples_fisheriris(X, Y, nsamp, RepFac);
            case 'crosssubj',
                [Xl Yl Xu Yu] = config_crosssubject(X, Y, subjects_all, subj_lab, subj_unlab);
            otherwise,
                error('run type unknown!');
        end
%         clear trees;
%         [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha);
        [trees acc] = sss(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha);
        acc_{i} = [acc_{i}; acc];
        toc;
    end
end

%accuracy over runs and subjects
figure;
for i=1:1,%size(subjects,1),
%     subplot(size(subjects,1)/5, 5, i);
    bar([acc_{i}(:,1), acc_{i}(:,end)]);
    imp(:,i) = (acc_{i}(:,end) - acc_{i}(:,1))./acc_{i}(:,1);
    ylim([.5 1]);
    xlim([0 size(acc_{1},1)+1]);
    set(gca,'FontSize',16)

end
savefig(sprintf('%s_accs_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)));
saveas(gcf,sprintf('%s_accs_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)),'png')

figure;
for i=1:1,%size(subjects,1),
%     subplot(size(subjects,1)/5, 5, i);
    imp_good = imp(:,i);
    imp_bad = imp(:,i);
    imp_good(imp_good<0) = 0;
    imp_bad(imp_bad>=0) = 0;
    bar(imp_good, 'g');
    hold on;
    bar(imp_bad, 'r');
    xlim([0 size(acc_{1},1)+2]);
    set(gca,'FontSize',14)
    ylim([-0.2 0.2])
end
savefig(sprintf('%s_imp_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)));
saveas(gcf,sprintf('%s_imp_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)),'png');


%improvement over different experiments
figure;
boxplot(imp);   set(gca,'FontSize',14)
savefig(sprintf('%s_imp_boxplot_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)));
saveas(gcf,sprintf('%s_imp_boxplot_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)),'png');

%overall improvement
figure;
boxplot(imp(:));     set(gca,'FontSize',14)
savefig(sprintf('%s_imp_boxplot_all_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)));
saveas(gcf,sprintf('%s_imp_boxplot_all_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)),'png');

%display accuracy at start and end for each subject
figure;
for i=1:1,%size(subjects,1),
%     subplot(size(subjects,1)/5, 5, i);
    boxplot([acc_{i}(:,1) acc_{i}(:,end)]);   %accuracy at start and end 
    set(gca,'FontSize',16)
    set(gca, 'xtick', [1 2], 'xticklabel', {'initial','final'});
    ylim([0.5 1])
end
savefig(sprintf('%s_acc_boxplot_all_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)));
saveas(gcf,sprintf('%s_acc_boxplot_all_ntrees%d_nsamp%d_nsub%d', simtype, ntrees, nsamp, size(subjects,2)),'png');