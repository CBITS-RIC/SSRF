%display improvements for different amount of labeled samples from single
%subjects

acc2 = load('acc_tr1_2samples.mat'); acc2 = acc2.acc;
acc4 = load('acc_tr1_4samples.mat'); acc4 = acc4.acc;
acc8 = load('acc_tr1_8samples.mat'); acc8 = acc8.acc;

%improvements
I2 = acc2(:,end,1)-acc2(:,1,1);
I4 = acc4(:,end,1)-acc4(:,1,1);
I8 = acc8(:,end,1)-acc8(:,1,1);

figure
boxplot([I2 I4 I8],'labels',{'2','4','8'})

%show original and final accuracy
acc = [acc(:,1,1) acc(:,end,1)];
figure('name','2 samples'), hold on
bar(acc), ylim([0.5 1])

acc2i = [acc2(:,1,1) acc2(:,end,1)];
figure('name','2 samples'), hold on
bar(acc2i), ylim([0.5 1])
acc4i = [acc4(:,1,1) acc4(:,end,1)];
figure('name','4 samples'), hold on
bar(acc4i), ylim([0.5 1])
acc8i = [acc8(:,1,1) acc8(:,end,1)];
figure('name','8 samples'), hold on
bar(acc8i), ylim([0.5 1])

