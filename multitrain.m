function multitrain

close all
clear all
N = 5;  %# of sims to run
epochs = 100;
F = cell(N,1);  %cell array containing all the trained forests
for k = 1:N
   
    F{k} = config;
    if k > N/2
        F{k}.tau = 50;
    end
    F{k}.trainforest(epochs);
    
end

save('trainedforests.mat','F')