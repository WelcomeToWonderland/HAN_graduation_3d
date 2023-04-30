path = 'D:\workspace\dataset\USCT\original\HR\50525.mat';
file = load(path);

% data = file('f1');
% class(data)

% % 定义三维坐标轴
% figure
% hax = gca;
% hax.Box = 'on';
% hax.BoxStyle = 'full';
% hax.Projection = 'perspective';
% hax.DataAspectRatio = [1 1 1];
% hax.XLim = [xmin xmax];
% hax.YLim = [ymin ymax];
% hax.ZLim = [zmin zmax];
% % 绘制三维图形
% plot3(x, y, z, 'b.');
% % 开启交互式旋转缩放
% rotate3d on
