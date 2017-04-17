from CubeEnv import *
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display

e = CubeEnv()
frames = []
for t in range(100):
    frames.append(e.render(mode="rgb_array"))
    action = e.action_space.sample()
    e.step(action)    
e.render(close=True)

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    fig=e.cube.show_layout(frames[0]) 
    print("Drawn")
    def animate(i):
        return e.cube.update_plot(frames[i])
    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50,blit=True)
    
display_frames_as_gif(frames)
plt.show()
