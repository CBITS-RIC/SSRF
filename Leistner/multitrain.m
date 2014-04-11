function F = multitrain

close all
clear all
epochs = 200;
acc = [];
%tau = [70];
trees = [5 20 40 80 160];
N = length(trees);    %# of simulations to run

F = cell(N,1);  %cell array containing all the trained forests
tic
for k = 1:N
   
    F{k} = config;
    
%     F{k}.tau = tau(k);
    F{k}.ntrees = trees(k);
    
    F{k}.trainforest_multic(epochs);
    
    acc = [acc;F{k}.acc];
    
end
t= toc

save('trainedforests_vartrees_4tr_4te_tau80.mat','F')

figure, hold on
plot(acc','LineWidth',2)
for s = 1:N
    l{s} = num2str(trees(s));
end

legend(l)