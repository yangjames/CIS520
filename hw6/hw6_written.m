theta=linspace(0,2*pi,360);
radius = 1/3;
circle_x = radius*cos(theta);
circle_y = radius*sin(theta);

X_circles=[-3 -2 3];
X_squares=[-1 0 1];

X2_circles = X_circles.^2;
X2_squares = X_squares.^2;

figure(1)
clf
hold on
grid on
axis equal
ylabel('X_1^2')
xlabel('X_1')
% plot circles
for i = 1:length(X_circles)
    plot(circle_x+X_circles(i),circle_y+X2_circles(i),'r-')
end

% plot squares
scale = 1/3;
square_x = scale*[1 -1 -1 1 1];
square_y = scale*[1 1 -1 -1 1];
for i = 1:length(X_squares)
    plot(square_x+X_squares(i),square_y+X2_squares(i),'b-')
end

% plot dividing line
vector = [-2 4]-[-1 1];
midpoint = vector/2+[-1 1];
slope = -vector(1)/vector(2);
x=-4:6;
y=slope*(x-midpoint(1))+midpoint(2);
plot(x,y,'k-')

% circle the support vectors
plot(circle_x*2-2, circle_y*2+4,'k-')
plot(circle_x*2-1, circle_y*2+1,'k-')

figure(2)
clf
hold on
grid on
axis equal
for i = 1:length(X_circles)
    plot(circle_x+X_circles(i),circle_y,'r-')
end
for i = 1:length(X_squares)
    plot(square_x+X_squares(i),square_y,'b-')
end
plot([-4 4],[0 0],'k-')
plot(circle_x*0.5+((1/3)+sqrt(1/9+12))/2,circle_y*0.5,'g-')
plot(circle_x*0.5+((1/3)-sqrt(1/9+12))/2,circle_y*0.5,'g-')