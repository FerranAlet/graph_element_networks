#!/usr/bin/env python

from manimlib.imports import *
from copy import copy, deepcopy
#from PIL import Image
#import requests
#from io import BytesIO

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.

SUPER_RED = rgb_to_hex(np.array([0.5,0.,0.]))
LIGHT_GREEN = average_color(GREEN, rgb_to_hex(np.array([0.,1.,0.])))
SUPER_GREEN = rgb_to_hex(np.array([0.,0.5,0.]))
SUPER_BLUE = rgb_to_hex(np.array([0.,0.,0.5]))
COLD_BLUE = rgb_to_hex(np.array([0.5,0.75,1.]))
np.random.seed(2)

class Fridge(SVGMobject):
  CONFIG = {
    "file_name": "fridge.svg",
  }
class Snowman(SVGMobject):
  CONFIG = {
    "file_name": "snowman.svg",
  }
class Cabinet(SVGMobject):
  CONFIG = {
    "file_name": "cabinet.svg",
  }
class Bed(SVGMobject):
  CONFIG = {
    "file_name": "bed.svg",
  }
def random_dot_in_square(sq, sz=0.03, color=SUPER_RED, buffer=0.05, color_fn=None):
  left, right, bottom, top = sq.get_left(), sq.get_right(), sq.get_bottom(), sq.get_top()
  x_rand = np.random.rand()
  y_rand = np.random.rand()
  x_r = x_rand*(1-2*buffer) + buffer
  y_r = y_rand*(1-2*buffer) + buffer
  x = (left + x_r*(right-left))[0]
  y = (bottom + y_r*(top-bottom))[1]
  rands = np.array([x_rand, y_rand,0])
  pos = np.array([x,y,0])
  if color == WHITE: color = BLACK
  if type(color) == str: color = hex_to_rgb(color)
  if color_fn is None: #random
    c = np.random.rand()
  else:
    c = color_fn(rands)
  c = rgb_to_hex((c*color + (1-c)*hex_to_rgb(WHITE)).astype(float))

  return Square(side_length=sz, color=c).move_to(pos)

def random_dots_in_square(sq, num, sz=0.03, color=SUPER_RED, buffer=0.05, color_fn=None, sort_pts = True):
  L = [random_dot_in_square(sq, sz=sz, color=color, buffer=buffer, color_fn=color_fn) for _ in range(num)]
  if sort_pts:
    L.sort(key=lambda x : x.get_center()[1])
  return L

def mesh_square(sq, num, sz=0.1, color=WHITE, fill_color=None, edge_color=None):
  if fill_color is None:
    fill_color = color
    opacity = 0
  else: opacity = 1
  if edge_color is None:
    edge_color = color
  vertices = []
  edges = []
  bot_left = sq.get_corner(BOTTOM+LEFT)
  dx, dy = (sq.get_right()-sq.get_left())/(num-1), (sq.get_top()-sq.get_bottom())/(num-1)
  for i in range(num):
    for j in range(num):
      col = color[i*num+j] if type(color) == type([]) else color
      fill_col = fill_color[i*num+j] if type(color) == type([]) else fill_color
      vertices.append(Circle(radius=sz,color=col, fill_color=fill_col).move_to(
        bot_left + i*dx + j*dy).set_fill(opacity=opacity))
      #Add edges to neighbours of less index
      if j:
        edges.append(Line(vertices[i*num+j-1].get_top(),
                          vertices[-1].get_bottom(), color=edge_color))
      if i:
        edges.append(Line(vertices[(i-1)*num+j].get_right(),
                          vertices[-1].get_left(), color=edge_color))

  return vertices, edges

def perturb_mesh(mesh_v, mesh_e, square, noise=0.05, bias=0.05, reuse_e=None):
  new_v = deepcopy(mesh_v)
  for v in new_v:
    v.shift(np.array([noise*np.random.normal(), noise*np.random.normal(), 0])+bias*DOWN+bias*RIGHT)
    v.move_to(np.maximum(square.get_corner(DOWN+LEFT), v.get_center()))
    v.move_to(np.minimum(square.get_corner(UP+RIGHT), v.get_center()))
  if reuse_e is None:
    new_e = [] ; edge_idx = []
    for i_a,a in enumerate(new_v):
      for i_b,b in enumerate(new_v[:i_a]):
        p = a.get_center()
        q = b.get_center()
        norm_pq = (q-p)/np.linalg.norm(p-q)
        m = (p+q)/2.
        for i_c, c in enumerate(new_v):
          if i_c!=i_a and i_c!=i_b and np.linalg.norm(c.get_center()-m) < np.linalg.norm(p-m): break
        else:
          new_e.append(Line(p+0.1*norm_pq, q-0.1*norm_pq, color=BLUE))
          edge_idx.append([i_a, i_b])
  else:
    edge_idx = reuse_e
    new_e = []
    for e in edge_idx:
      p = new_v[e[0]].get_center() ; q = new_v[e[1]].get_center()
      norm_pq = (q-p)/np.linalg.norm(p-q)
      new_e.append(Line(p+0.1*norm_pq, q-0.1*norm_pq, color=BLUE))

  return new_v, new_e, edge_idx

def change_random_colors(L, ori=SUPER_RED, dest=SUPER_BLUE, base=WHITE, new_base=None):
  if new_base is None:
    new_base = base
    for l in L:
      num = np.max(np.abs((color_to_rgb(l.get_color())-hex_to_rgb(base))/(hex_to_rgb(ori)-hex_to_rgb(base)+1e-3)))
      color = rgb_to_hex(num*hex_to_rgb(dest) + (1-num)*hex_to_rgb(new_base))
      l.set_color(color)


def keep_next_to(obj, reference, relation):
  obj.next_to(reference, relation)



def sample_heaters(sq):
  sz=0.06
  points = []
  colors = [SUPER_RED, RED]
  #rxs = [0.15, 0.8]
  #rys = [0.3, 0.6]
  centers = [np.array([0.15,0.8,0]), np.array([0.3,0.6,0.])]

  dx = [-1,0,1]; dy = [-1,0,1]
  for h in range(2):
    centers[h] = sq.get_corner(BOTTOM+LEFT)+centers[h]*np.array([sq.get_width(), sq.get_height(), 0])
    for i in range(3):
      for j in range(3):
        points.append(Square(color=colors[h], side_length=sz).move_to(
          centers[h]+0.1*dx[i]*RIGHT+0.1*dy[j]*UP).set_fill(opacity=1))
  return points

def sample_boundary(sq, color=COLD_BLUE, num_per_side=4, sz=0.1):
  points = []
  for i in range(num_per_side):
    points.append(Square(color=color, side_length=sz).move_to(
      sq.get_corner(TOP+LEFT) + i*sq.get_width()/num_per_side*RIGHT).set_fill(opacity=1))
  for i in range(num_per_side):
    points.append(Square(color=color, side_length=sz).move_to(
      sq.get_corner(TOP+RIGHT) + i*sq.get_height()/num_per_side*DOWN).set_fill(opacity=1))
  for i in range(num_per_side):
    points.append(Square(color=color, side_length=sz).move_to(
      sq.get_corner(BOTTOM+RIGHT) + i*sq.get_width()/num_per_side*LEFT).set_fill(opacity=1))
  for i in range(num_per_side):
    points.append(Square(color=color, side_length=sz).move_to(
      sq.get_corner(BOTTOM+LEFT) + i*sq.get_width()/num_per_side*UP).set_fill(opacity=1))
  return points

def get_img_obj_from_url(url):
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))


class ICML(Scene):
  def new_slide(self, num):
    self.slide = num
    self.time_factor = 1.5 if self.slide in self.slides else self.eps_time
    if num == 1:
      self.slide_text = TextMobject(str(self.slide))
      self.slide_pos = BOTTOM+0.5*UP
      self.add(self.slide_text.move_to(self.slide_pos))
    else:
      self.wait(1.5)
      new_slide_text = TextMobject(str(self.slide)).move_to(self.slide_pos)
      self.play(Transform(self.slide_text, new_slide_text),
                run_time=0.5*self.time_factor)

  def representation_function_process(self, outer_sq, mesh_v, mesh_e, dots, interpolate=False, do_play=True):
    func_factor = 1.
    temp_arrows = []
    num = np.round(np.sqrt(len(mesh_v)))
    dx, dy = (outer_sq.get_right()-outer_sq.get_left())/(num-1), (outer_sq.get_top()-outer_sq.get_bottom())/(num-1)
    mesh_colors = [[] for _ in mesh_v]
    for i,sq in enumerate(dots):
      sq_i = (sq.get_center()-outer_sq.get_left())[0]/dx[0]
      sq_j = (sq.get_center()-outer_sq.get_bottom())[1]/dy[1]
      if interpolate:
        temp_arrows += [
          Line(mesh_v[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)].get_corner(TOP+RIGHT), sq.get_bottom()), #Bottom-Left
          Line(mesh_v[np.round(np.floor(sq_j)+np.ceil(sq_i)*num).astype(int)].get_corner(TOP+LEFT), sq.get_right()), #Bottom-Right
          Line(mesh_v[np.round(np.ceil(sq_j)+np.floor(sq_i)*num).astype(int)].get_corner(BOTTOM+RIGHT), sq.get_top()), #Top-Left
          Line(mesh_v[np.round(np.ceil(sq_j)+np.ceil(sq_i)*num).astype(int)].get_corner(BOTTOM+LEFT), sq.get_left()) #Top-Right
      ]
      else:
        temp_arrows += [
          Line(sq.get_bottom(), mesh_v[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)].get_corner(TOP+RIGHT)), #Bottom-Left
          Line(sq.get_right(), mesh_v[np.round(np.floor(sq_j)+np.ceil(sq_i)*num).astype(int)].get_corner(TOP+LEFT)), #Bottom-Right
          Line(sq.get_top(), mesh_v[np.round(np.ceil(sq_j)+np.floor(sq_i)*num).astype(int)].get_corner(BOTTOM+RIGHT)), #Top-Left
          Line(sq.get_left(), mesh_v[np.round(np.ceil(sq_j)+np.ceil(sq_i)*num).astype(int)].get_corner(BOTTOM+LEFT)) #Top-Right
        ]
      if do_play:
        self.play(
          *[GrowArrow(_) for _ in temp_arrows[-4:]],
          run_time=1*self.time_factor*func_factor/(i+1))
      mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)].append(sq.get_color())
      mesh_colors[np.round(np.floor(sq_j)+np.ceil(sq_i)*num).astype(int)].append(sq.get_color())
      mesh_colors[np.round(np.ceil(sq_j)+np.floor(sq_i)*num).astype(int)].append(sq.get_color())
      mesh_colors[np.round(np.ceil(sq_j)+np.ceil(sq_i)*num).astype(int)].append(sq.get_color())
      if interpolate and do_play:
        self.play(
          FadeIn(sq),
          *[FadeOut(_) for _ in temp_arrows[-4:]],
          run_time=0.5* self.time_factor*func_factor/(i+1))
      elif do_play:
        self.play(
          ApplyMethod(mesh_v[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)].set_color,
                     average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)])),
          ApplyMethod(mesh_v[np.round(np.floor(sq_j)+np.ceil(sq_i)*num).astype(int)].set_color,
                     average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)])),
          ApplyMethod(mesh_v[np.round(np.ceil(sq_j)+np.floor(sq_i)*num).astype(int)].set_color,
                     average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)])),
          ApplyMethod(mesh_v[np.round(np.ceil(sq_j)+np.ceil(sq_i)*num).astype(int)].set_color,
                     average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)])),
          *[FadeOut(_) for _ in temp_arrows[-4:]],
          run_time=0.5* self.time_factor*func_factor/(i+1))
      else:
        mesh_v[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)].set_color(
          average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)]))
        mesh_v[np.round(np.floor(sq_j)+np.ceil(sq_i)*num).astype(int)].set_color(
          average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)]))
        mesh_v[np.round(np.ceil(sq_j)+np.floor(sq_i)*num).astype(int)].set_color(
          average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)]))
        mesh_v[np.round(np.ceil(sq_j)+np.ceil(sq_i)*num).astype(int)].set_color(
          average_color(*mesh_colors[np.round(np.floor(sq_j)+np.floor(sq_i)*num).astype(int)]))

  def message_passing(self, mesh_v, mesh_e, updt_color=YELLOW, steps=1, final_mesh=None):
    for step in range(steps):
      #Message accross edges
      edge_pts = []
      for e in mesh_e:
        edge_pts.append(Line(e.get_start(), e.get_start()*0.85+e.get_end()*0.15, color=updt_color, stroke_width=4))
        edge_pts.append(Line(e.get_end(), e.get_start()*0.15+e.get_end()*0.85, color=updt_color, stroke_width=4))
      edge_pts_copy = deepcopy(edge_pts)
      self.play(
        *[Transform(pt, edge_pts_copy[i+1-2*(i%2)]) for (i, pt) in enumerate(edge_pts)],
        run_time=1.5*self.time_factor)
      self.play(
        *[FadeOut(pt) for (i, pt) in enumerate(edge_pts)],
        run_time=1*self.time_factor)
      node_updt = [v.copy().set_color(updt_color).set_fill(opacity=0) for v in mesh_v]
      self.play(
        *[FadeIn(v) for v in node_updt],
        run_time=0.5*self.time_factor)
      self.play(
        #*[Transform(v, v.copy().set_color(rgb_to_hex(np.minimum(np.array([1,1,1]), np.array(1,1,1))))) #hex_to_rgb(v.get_color())*np.random.rand()*2)))
         # for v in mesh_v],
        *[Transform(v, v.copy().set_color(average_color(v.get_color(),
                                                        rgb_to_hex(np.random.rand()*hex_to_rgb(WHITE)) if final_mesh is None
                                                                  else final_mesh[i])))
          for i,v in enumerate(mesh_v)],
        *[FadeOut(updt) for updt in node_updt],
        run_time=0.5*self.time_factor)

  def construct(self):
    self.slides = [2018]#[1,2,-2,3,4,5,6,7,8,1781]
    self.eps_time = eps_time = 0.05
    self.slide_dict = {
      1 : 'FunctionToFunction',
      2 : 'LatentFunctionAppears',
      -2: 'Two inputs and two outputs',
      3 : 'Two latent functions',
      4 : 'Creating a mesh',
      5: 'Three square transition',
      6: 'Message passing',
      7: 'Interpolate and decode',
      1781 : 'Poissson',
      2018 : 'Neural Scene Representation'
    }
    # 1: FunctionToFunction
    self.new_slide(1)
    #########################
    # 1: FunctionToFunction #
    #########################
    space_square = Square(color=WHITE).move_to(np.array([0,3,0]))
    inp_square = Square(color=RED).move_to(np.array([-2,0,0]))
    inp_square_text = TextMobject('Input\\\\samples', color=inp_square.get_color()).next_to(inp_square, DOWN)
    out_square = Square(color=GREEN).move_to(np.array([2,0,0]))
    out_square_text = TextMobject('Output\\\\queries', color=out_square.get_color()).next_to(out_square, DOWN)
    self.add(space_square,inp_square, out_square, inp_square_text, out_square_text)
    space_to_inp = Line(space_square.get_left()+DOWN*space_square.get_height()/2., inp_square.get_top()).add_tip()
    space_to_out = Line(space_square.get_right()+DOWN*space_square.get_height()/2., out_square.get_top()).add_tip()
    p_a = random_dot_in_square(inp_square)
    p_b = random_dot_in_square(inp_square)
    inp_points = random_dots_in_square(inp_square, 20, color_fn = lambda x : min(1.,1.2*np.linalg.norm(x-np.array([0.75,0.75,0])).item())) #np.min(1.,1.5*np.linalg.norm(x-np.array([0.25,0.5,0]))))
    inp_group = VGroup(inp_square, inp_square_text, *inp_points)
    inp_point_creations = [ShowCreation(pt) for pt in inp_points]
    out_points = random_dots_in_square(out_square, 20, color=SUPER_GREEN, color_fn = lambda x : min(1.,1.2*np.linalg.norm(x-np.array([0.25,0.75,0])).item()))
    out_point_creations = [ShowCreation(pt) for pt in out_points]
    out_group = VGroup(out_square, out_square_text, *out_points)
    self.play(
      GrowArrow(space_to_inp),
      GrowArrow(space_to_out),
      run_time=1 if 1 in self.slides else eps_time
    )
    self.play(
      *inp_point_creations,
      *out_point_creations,
      run_time=0.3*self.time_factor
    )
    inp_out_arrow = Line(inp_square.get_right(), out_square.get_left()).add_tip().set_color(RED,GREEN).set_sheen_direction(RIGHT)
    self.play(
      ShowCreation(inp_out_arrow),
      FadeOut(space_square),
      FadeOut(space_to_inp),
      FadeOut(space_to_out),
      run_time=1.5 if 1 in self.slides else eps_time
    )
    #############################
    # 2 : LatentFunctionAppears #
    #############################
    self.new_slide(2)
    self.play(
      ApplyMethod(inp_group.shift, 4*LEFT),
      ApplyMethod(out_group.shift, 4*RIGHT),
      #*[ApplyMethod(_.shift, 4*LEFT) for _ in [inp_square, *inp_points]],
      #*[ApplyMethod(_.shift, 4*RIGHT) for _ in [out_square, *out_points]],
      FadeOut(inp_out_arrow),
      run_time=2*self.time_factor
    )
    #Input to latent
    lat_square = Square(color=BLUE)
    lat_square_text = TextMobject('Latent\\\\function',color=BLUE).next_to(lat_square, DOWN)
    inp_lat_arrow = Line(inp_square.get_right(), lat_square.get_left()).add_tip().set_color(color=[RED,BLUE]).set_sheen_direction(RIGHT)
      #RED,RED)
      #inp_square.get_color(), lat_square.get_color())
    inp_lat_arrow_text = TextMobject('Encode').next_to(inp_lat_arrow, TOP).set_color_by_gradient(RED,BLUE)
    self.play(
      FadeIn(lat_square, lat_square_text),
      run_time=2*self.time_factor
    )
    inp_lat_points = [deepcopy(_).shift(lat_square.get_center()-inp_square.get_center()) for _ in inp_points]
    change_random_colors(inp_lat_points,SUPER_RED, SUPER_BLUE)
    inp_lat_point_arrows = [Line(inp_pt, lat_pt).add_tip().set_color(
      inp_square.get_color(), out_square.get_color()).set_sheen_direction(RIGHT) for (inp_pt, lat_pt) in zip(inp_points, inp_lat_points)]
    self.play(
      *[GrowArrow(_) for _ in inp_lat_point_arrows],
      GrowArrow(inp_lat_arrow),
      run_time=2*self.time_factor
    )
    self.play(
      *[FadeOut(_) for _ in inp_lat_point_arrows],
      *[FadeIn(_) for _ in inp_lat_points+[inp_lat_arrow_text]],
      run_time=.5*self.time_factor
    )
    #Latent to output
    out_lat_points = [deepcopy(_).shift(lat_square.get_center()-out_square.get_center()) for _ in out_points]
    change_random_colors(out_lat_points,SUPER_GREEN, SUPER_BLUE)
    out_lat_point_arrows = [Line(lat_pt, out_pt).add_tip().set_color(
      lat_square.get_color(), out_square.get_color()).set_sheen_direction(RIGHT) for (lat_pt, out_pt)
                            in zip(out_lat_points, out_points)]
    out_lat_arrow = Line(lat_square.get_right(), out_square.get_left()).add_tip().set_color(color=[BLUE,GREEN]).set_sheen_direction(RIGHT)
    out_lat_arrow_text = TextMobject('Decode').next_to(out_lat_arrow, TOP).set_color_by_gradient(BLUE,GREEN)

    self.play(
      *[FadeOut(_) for _ in out_points],
      *[FadeIn(_) for _ in out_lat_points],
      run_time=1*self.time_factor
    )
    self.play(
      *[GrowArrow(_) for _ in out_lat_point_arrows],
      GrowArrow(out_lat_arrow),
      run_time=2*self.time_factor
    )
    self.play(
      *[FadeOut(_) for _ in out_lat_point_arrows],
      *[FadeIn(_) for _ in out_points+[out_lat_arrow_text]],
      run_time=.5*self.time_factor
    )
    ##################################
    # -2: Two inputs and two outputs #
    ##################################
    sec_inp_square = inp_square.copy().set_color(ORANGE).next_to(inp_square, DOWN)
    sec_out_square = out_square.copy().set_color(YELLOW).next_to(out_square, DOWN)
    sec_inp_points = random_dots_in_square(sec_inp_square, 20, color=ORANGE, color_fn = lambda x : min(1.,2.*np.abs(x[0]-0.5)).item())
    sec_out_points = random_dots_in_square(sec_out_square, 20, color=YELLOW, color_fn = lambda x : min(1.,np.abs(np.cos(x[0]+2.*x[1]))).item())
    sec_inp_arrow = Line(sec_inp_square.get_right(), lat_square.get_left()).add_tip().set_color(color=[ORANGE,BLUE]).set_sheen_direction(RIGHT)
    sec_out_arrow = Line(lat_square.get_right(), sec_out_square.get_left()).add_tip().set_color(color=[BLUE, YELLOW]).set_sheen_direction(RIGHT)
    self.play(
      FadeOut(inp_square_text), FadeOut(out_square_text),
      FadeIn(sec_inp_square), FadeIn(sec_out_square),
      *[FadeIn(_) for _ in sec_inp_points+sec_out_points],
      run_time = 2 * self.time_factor
    )
    self.play(
      GrowArrow(sec_inp_arrow), GrowArrow(sec_out_arrow),
      run_time = 3 * self.time_factor
    )
    self.play(
      FadeIn(inp_square_text), FadeIn(out_square_text),
      FadeOut(sec_inp_square), FadeOut(sec_out_square),
      FadeOut(sec_inp_arrow), FadeOut(sec_out_arrow),
      *[FadeOut(_) for _ in sec_inp_points+sec_out_points],
      run_time = 2 * self.time_factor
    )
    ###########################
    # 3: Two latent functions #
    ###########################
    self.new_slide(3)
    inp_lat_square = lat_square.copy().set_color(BLUE)
    out_lat_square = lat_square.copy().set_color(BLUE)
    self.play(
      FadeIn(inp_lat_square),
      FadeIn(out_lat_square),
      FadeOut(lat_square, lat_square_text),
      run_time=1*self.time_factor
    )
    final_inp_lat_arrow = inp_lat_arrow.copy().add_tip()
    final_inp_lat_arrow.put_start_and_end_on(inp_lat_arrow.get_start(), lat_square.get_left()+2*LEFT)
    final_out_lat_arrow = out_lat_arrow.copy().add_tip()
    final_out_lat_arrow.put_start_and_end_on(lat_square.get_right()+2*RIGHT, out_lat_arrow.get_end())
    self.play(
      *[ApplyMethod(_.shift, 2*LEFT) for _ in [inp_lat_square, *inp_lat_points]],
      *[ApplyMethod(_.shift, 2*RIGHT) for _ in [out_lat_square, *out_lat_points]],
      Transform(inp_lat_arrow, final_inp_lat_arrow),
      Transform(out_lat_arrow, final_out_lat_arrow),
      UpdateFromFunc(inp_lat_arrow_text, lambda text : keep_next_to(text, inp_lat_arrow, TOP)),
      UpdateFromFunc(out_lat_arrow_text, lambda text : keep_next_to(text, out_lat_arrow, TOP)),
      #Transform(out_lat_arrow, Arrow(lat_square.get_right()+2*RIGHT, out_lat_arrow.get_end(),
      #                               color=out_lat_arrow.get_color(), buff=0)),
      #.set_color_by_gradient(out_lat_square.get_color(),out_square.get_color())
      run_time=3*self.time_factor
    )
    inp_lat_square_text = TextMobject("""Inp. latent\\\\samples""",
                                      color=inp_lat_square.get_color()).next_to(inp_lat_square, DOWN)
    inp_lat_group = VGroup(inp_lat_square, inp_lat_square_text, *inp_lat_points)
    out_lat_square_text = TextMobject("""Out. latent\\\\queries""",
                                      color=out_lat_square.get_color()).next_to(out_lat_square, DOWN)
    out_lat_group = VGroup(out_lat_square, out_lat_square_text, *out_lat_points)
    self.play(
      #*[FadeOut(_) for _ in out_lat_points],
      *[FadeIn(_) for _ in [inp_lat_square_text, out_lat_square_text]],
      run_time=1.*self.time_factor
    )
    ######################
    # 4: Creating a mesh #
    ######################
    self.new_slide(4)
    inp_lat_mesh_v, inp_lat_mesh_e = mesh_square(inp_lat_square, 4, color=inp_lat_square.get_color(),
                                                 fill_color=inp_lat_square.get_color())
    out_lat_mesh_v, out_lat_mesh_e = mesh_square(out_lat_square, 4, color=out_lat_square.get_color(),
                                                 fill_color=out_lat_square.get_color())
    next_inp_lat_square_text = TextMobject('Inp. latent\\\\function', color=inp_lat_square_text.get_color()).next_to(
      inp_lat_square, DOWN)
    next_out_lat_square_text = TextMobject('Out. latent\\\\function', color=out_lat_square_text.get_color()).next_to(
      out_lat_square, DOWN)
    self.play(
      *[FadeIn(_) for _ in inp_lat_mesh_v+inp_lat_mesh_e+out_lat_mesh_v+out_lat_mesh_e],
      Transform(inp_lat_square_text, next_inp_lat_square_text),
      Transform(out_lat_square_text, next_out_lat_square_text),
      run_time=1*self.time_factor
    )
    lat_lat_arrow = Arrow(inp_lat_square.get_right(), out_lat_square.get_left(), color=BLUE)
    lat_lat_arrow_text = TextMobject('Message\\\\Passing',
                                     color=lat_lat_arrow.get_color()).next_to(lat_lat_arrow, UP, buff=1)
    self.play(
      FadeIn(lat_lat_arrow),
      FadeIn(lat_lat_arrow_text),
      run_time=1*self.time_factor)
    ##############################
    # 5: Three square transition #
    ##############################
    self.new_slide(5)
    self.play(
      FadeOut(lat_lat_arrow), FadeOut(lat_lat_arrow_text),
      *[FadeOut(aux) for aux in out_lat_mesh_v+out_lat_mesh_e],
      FadeOut(out_lat_group),
      Transform(inp_lat_square_text, TextMobject('Latent function').set_color(BLUE).next_to(inp_lat_square, DOWN)),
      run_time=2*self.time_factor
    )
    #First compute the final color in hindsight
    self.representation_function_process(out_lat_square, out_lat_mesh_v, out_lat_mesh_e, out_lat_points, do_play=False)
    final_mesh_colors = [_.get_color() for _ in out_lat_mesh_v]
    for v in out_lat_mesh_v: v.set_color(BLUE)
    start_mesh_colors = [_.get_color() for _ in inp_lat_mesh_v]
    for i,v in enumerate(out_lat_mesh_v):
      v.set_color(final_mesh_colors[i])
    ori_out_lat_arrow = out_lat_arrow.copy()
    new_out_lat_arrow = out_lat_arrow.copy().add_tip()
    new_out_lat_arrow.put_start_and_end_on(np.array([2,0,0]), out_square.get_left())
    final_inp_lat_square = inp_lat_square.copy().move_to(np.array([0,0,0]))#.set_height(4).set_width(4)
    final_inp_lat_mesh_v, final_inp_lat_mesh_e = mesh_square(final_inp_lat_square, 4, color=start_mesh_colors,
                                                             fill_color=start_mesh_colors, edge_color=inp_lat_square.get_color())
    self.play(
      Transform(inp_lat_square, final_inp_lat_square),
      *[ApplyMethod(_.shift, final_inp_lat_square.get_center()-inp_lat_square.get_center()) for _ in inp_lat_points],
      UpdateFromFunc(inp_lat_square_text, lambda txt : txt.next_to(inp_lat_square, DOWN)),
      UpdateFromFunc(inp_lat_arrow, lambda arr : arr.put_start_and_end_on(inp_square.get_right(), inp_lat_square.get_left())),
      UpdateFromFunc(inp_lat_arrow_text, lambda txt : txt.next_to(inp_lat_arrow, TOP)),
      #Transform(out_lat_arrow, new_out_lat_arrow),
      UpdateFromFunc(out_lat_arrow, lambda arr : arr.put_start_and_end_on(inp_lat_square.get_right(), out_square.get_left())),
      UpdateFromFunc(out_lat_arrow_text, lambda txt : txt.next_to(out_lat_arrow, TOP)),
      *[Transform(a,b) for (a,b) in zip(inp_lat_mesh_v+inp_lat_mesh_e, final_inp_lat_mesh_v + final_inp_lat_mesh_e)],
      run_time=3*self.time_factor
    )
    self.wait(0.5)
    self.representation_function_process(inp_lat_square, inp_lat_mesh_v, inp_lat_mesh_e, inp_lat_points)
    self.play(
      *[FadeOut(pt) for pt in inp_lat_points],
      run_time=1*self.time_factor)
    #final_inp_square = inp_square.copy().set_width(inp_square.get_width()*1.5).set_height(inp_square.get_height()*1.5).shift(0.5*RIGHT)
    #final_inp_square = out_square.copy().set_width(out_square.get_width()*1.5).set_height(out_square.get_height()*1.5).shift(0.5*LEFT)
    ######################
    # 6: Message passing #
    ######################
    self.new_slide(6)
    #Move and make GNN_mesh bigger
    #self.message_passing(GNN_mesh_v, GNN_mesh_e, steps=3, final_mesh=final_mesh_colors)
    self.message_passing(inp_lat_mesh_v, inp_lat_mesh_e, steps=4, final_mesh=final_mesh_colors)
    for v in out_lat_points:
      v.shift(inp_lat_square.get_center()-out_lat_square.get_center())
    #############################
    # 7: Interpolate and decode #
    #############################
    self.representation_function_process(inp_lat_square, inp_lat_mesh_v, inp_lat_mesh_e, out_lat_points, interpolate=True)
    out_lat_arrows = [Arrow(out_lat_pt, out_pt).set_color(BLUE,GREEN).add_tip().set_color(
        color=[out_lat_square.get_color(),out_square.get_color()]).set_sheen_direction(RIGHT)
                      for (out_lat_pt, out_pt) in zip(out_lat_points, out_points)]
    self.play(
      *[GrowArrow(_) for _ in out_lat_arrows],
      run_time=1.5*self.time_factor,
    )
    self.play(
      *[FadeOut(_) for _ in out_lat_arrows],
      run_time=1*self.time_factor,
    )
    ######################
    # 8: Parametrization #
    ######################
    self.new_slide(8)
    enc_sym = TexMobject('NN_{\\theta_{enc}}',
                           color=YELLOW).scale(0.75).next_to(inp_lat_arrow_text, DOWN)
    dec_sym = TexMobject('NN_{\\theta_{dec}}',
                           color=YELLOW).scale(0.75).next_to(out_lat_arrow_text, DOWN)
    self.play(
      FadeIn(enc_sym),
      run_time=1*self.time_factor)
    self.play(
      FadeIn(dec_sym),
      run_time=1*self.time_factor)
    gnn_text = TextMobject('Graph NN').move_to(np.array([-5.5,3.5,0])).set_color(BLUE)
    edgenn_text = TextMobject('Edge NN').next_to(gnn_text, RIGHT, buff=0.5).set_color(BLUE)
    nodenn_text = TextMobject('Node NN').next_to(edgenn_text, DOWN).set_color(BLUE)
    edgenn_sym = inp_lat_mesh_e[-1].copy().set_color(YELLOW).next_to(edgenn_text, RIGHT)
    nodenn_sym = inp_lat_mesh_v[0].copy().set_color(YELLOW).set_fill(opacity=0).next_to(nodenn_text, RIGHT)
    nodenn_sym.shift((edgenn_sym.get_center()-nodenn_sym.get_center())*RIGHT)
    edgenn_eq = TexMobject('NN_{\\theta_e}(h^t_w, h^t_v) \\rightarrow msg(w,v)',
                           color=YELLOW).scale(0.75).next_to(edgenn_sym, RIGHT)
    nodenn_eq = TexMobject('NN_{\\theta_n}(h^t_v,',
                           color=YELLOW).scale(0.75).next_to(nodenn_sym, RIGHT)
    nodenn_eq.shift((edgenn_eq.get_left()-nodenn_eq.get_left())*RIGHT)
    pool_eq = TexMobject('\\sum_{w\in neigh(v)} msg(w,v)',
                         color=YELLOW).scale(0.5).next_to(nodenn_eq, RIGHT)
    nodenn_eq_end =TexMobject(') \\rightarrow h_v^{t+1}', color=YELLOW).scale(0.75).next_to(pool_eq, RIGHT)
    pool_eq.shift(0.1*DOWN)
    self.play(
      *[FadeIn(_) for _ in [gnn_text, edgenn_text, edgenn_sym]],
      run_time=2*self.time_factor)
    self.play(
      *[FadeIn(_) for _ in [nodenn_text, nodenn_sym]],
      run_time=2*self.time_factor)
    self.play(
      *[FadeIn(_) for _ in [edgenn_eq]],
      run_time=2*self.time_factor)
    self.play(
      *[FadeIn(_) for _ in [pool_eq]],
      run_time=2*self.time_factor)
    self.play(
      *[FadeIn(_) for _ in [nodenn_eq, nodenn_eq_end]],
      run_time=2*self.time_factor)

    ###################################
    # 9: Combinatorial generalization #
    ###################################
    self.new_slide(9)
    #Copy all edges and all nodes, turn them yellow and move them to the original position
    yellow_inp_lat_mesh_v = [nodenn_sym.deepcopy() for _ in inp_lat_mesh_v]
    yellow_inp_lat_mesh_e = [edgenn_sym.deepcopy() for _ in inp_lat_mesh_e]
    yellow_inp_lat_mesh_v_end, yellow_inp_lat_mesh_e_end = deepcopy(inp_lat_mesh_v), deepcopy(inp_lat_mesh_e)
    for aux in yellow_inp_lat_mesh_v_end: aux.set_color(YELLOW).set_fill(opacity=0)
    for aux in yellow_inp_lat_mesh_e_end: aux.set_color(YELLOW)
    self.add(*yellow_inp_lat_mesh_v, *yellow_inp_lat_mesh_e)
    self.play(
      *[Transform(a,b) for (a,b) in zip(yellow_inp_lat_mesh_e,yellow_inp_lat_mesh_e_end)],
      run_time=3*self.time_factor)
    self.play(
      *[Transform(a,b) for (a,b) in zip(yellow_inp_lat_mesh_v,yellow_inp_lat_mesh_v_end)],
      run_time=3*self.time_factor)
    self.play(
      *[FadeOut(a) for a in yellow_inp_lat_mesh_v+yellow_inp_lat_mesh_e],
      run_time=1*self.time_factor)
    ################################
    # 10: Different mesh densities #
    ################################
    self.new_slide(10)
    inp_lat_mesh_v_3, inp_lat_mesh_e_3 = mesh_square(inp_lat_square, 3, color=inp_lat_square.get_color(),
                                                 fill_color=inp_lat_square.get_color())
    self.representation_function_process(inp_lat_square, inp_lat_mesh_v_3, inp_lat_mesh_e_3, out_lat_points, do_play=False)
    inp_lat_mesh_v_5, inp_lat_mesh_e_5 = mesh_square(inp_lat_square, 5, color=inp_lat_square.get_color(),
                                                 fill_color=inp_lat_square.get_color())
    self.representation_function_process(inp_lat_square, inp_lat_mesh_v_5, inp_lat_mesh_e_5, out_lat_points, do_play=False)
    self.play(
      *[FadeOut(a) for a in inp_lat_mesh_v+inp_lat_mesh_e],
      *[FadeIn(a) for a in inp_lat_mesh_v_3+inp_lat_mesh_e_3],
      run_time=1*self.time_factor)
    self.wait(4*self.time_factor)
    self.play(
      *[FadeOut(a) for a in inp_lat_mesh_v_3+inp_lat_mesh_e_3],
      *[FadeIn(a) for a in inp_lat_mesh_v_5+inp_lat_mesh_e_5],
      run_time=1*self.time_factor)
    self.wait(4*self.time_factor)
    self.play(
      *[FadeOut(a) for a in inp_lat_mesh_v_5+inp_lat_mesh_e_5],
      *[FadeIn(a) for a in inp_lat_mesh_v+inp_lat_mesh_e],
      run_time=1*self.time_factor)
    ######################
    # 11: Node positions #
    ######################
    self.new_slide(11)
    nodepos_text = TextMobject('Node pos', color=BLUE).next_to(nodenn_text, DOWN)
    nodepos_sym = TexMobject('+', fill_color=YELLOW).next_to(nodepos_text, RIGHT)
    self.play(
      FadeIn(nodepos_text),
      FadeIn(nodepos_sym),
      run_time=1*self.time_factor)
    self.wait(1*self.time_factor)
    inp_lat_mesh_v_first, inp_lat_mesh_e_first, edge_idx = perturb_mesh(inp_lat_mesh_v, inp_lat_mesh_e, inp_lat_square, noise=0.1, bias=0.)
    self.play(
      *[FadeOut(a) for a in inp_lat_mesh_v+inp_lat_mesh_e],
      *[FadeIn(a) for a in inp_lat_mesh_v_first+inp_lat_mesh_e_first],
      run_time=1*self.time_factor)
    inp_lat_mesh_v_second, inp_lat_mesh_e_second, edge_idx = perturb_mesh(inp_lat_mesh_v_first, inp_lat_mesh_e_first,
                                                                          inp_lat_square, reuse_e=edge_idx)
    self.play(
      *[Transform(a,b) for (a,b) in zip(inp_lat_mesh_v_first+inp_lat_mesh_e_first,inp_lat_mesh_v_second+inp_lat_mesh_e_second)],
      run_time=2*self.time_factor)
    self.wait(2*self.time_factor)
    inp_lat_mesh_v_third, inp_lat_mesh_e_third, edge_idx = perturb_mesh(inp_lat_mesh_v_second, inp_lat_mesh_e_second, inp_lat_square, reuse_e=edge_idx)
    self.play(
      *[Transform(a,b) for (a,b) in zip(inp_lat_mesh_v_first+inp_lat_mesh_e_first, inp_lat_mesh_v_third+inp_lat_mesh_e_third)],
      run_time=2*self.time_factor)
    inp_lat_mesh_v_fourth, inp_lat_mesh_e_fourth, edge_idx = perturb_mesh(inp_lat_mesh_v_third, inp_lat_mesh_e_third, inp_lat_square, reuse_e=edge_idx)
    self.play(
      *[Transform(a,b) for (a,b) in zip(inp_lat_mesh_v_first+inp_lat_mesh_e_first, inp_lat_mesh_v_fourth+inp_lat_mesh_e_fourth)],
      run_time=2*self.time_factor)
    self.wait(2*self.time_factor)
    self.play(
      *[FadeIn(a) for a in inp_lat_mesh_v+inp_lat_mesh_e],
      *[FadeOut(a) for a in inp_lat_mesh_v_first+inp_lat_mesh_e_first],
      run_time=1*self.time_factor)
    #Reset basic scene
    self.remove(gnn_text, edgenn_text, edgenn_sym, nodenn_text, nodenn_sym,
                edgenn_eq, pool_eq, nodenn_eq, nodenn_eq_end, nodepos_text, nodepos_sym,
                enc_sym, dec_sym, *inp_lat_points, *out_lat_points)
    for v in inp_lat_mesh_v: v.set_color(BLUE)
    ##########################
    # 1781: Poisson equation #
    ##########################
    if 1781 in self.slides:
      self.new_slide(1781)
      #img = ImageMobject('snowman.svg')
      #self.play(ShowCreation(img))
      boun_inp_square = inp_square.copy().set_color(WHITE)
      inp_points_boundary = sample_boundary(inp_square, color=COLD_BLUE, num_per_side = 4, sz=0.1)
      boun_inp_square_text = TextMobject('Boundary\\\\conditions', color=boun_inp_square.get_color()).next_to(boun_inp_square, UP)
      heater_text = TextMobject('Laplacian', color=sec_inp_square.get_color()).next_to(sec_inp_square, DOWN)
      self.play(
        FadeIn(sec_inp_square),
        FadeOut(inp_group),
        FadeIn(boun_inp_square),
        run_time=1*self.time_factor)
      self.play(
        FadeIn(boun_inp_square_text),
        run_time=1*self.time_factor)
      self.play(
        FadeIn(heater_text),
        run_time=1*self.time_factor)
      inp_points_heater = sample_heaters(sec_inp_square)
      self.play(
        #*[FadeOut(_) for _ in inp_points],
        *[FadeIn(_) for _ in inp_points_heater+inp_points_boundary],
        run_time=3*self.time_factor
      )
      self.wait(2*self.time_factor)
    ##########################################
    # 2018: Scene representation experiments #
    ##########################################
    self.new_slide(2018)
    # Turn input and output white
    white_inp_square = inp_square.copy().set_color(WHITE)
    white_inp_square_text = inp_square_text.copy().set_color(WHITE)
    white_out_square = out_square.copy().set_color(WHITE)
    white_out_square_text = out_square_text.copy().set_color(WHITE)
    white_inp_lat_arrow = inp_lat_arrow.copy().set_color([WHITE, BLUE])
    white_out_lat_arrow = out_lat_arrow.copy().set_color([WHITE, BLUE])
    white_inp_lat_arrow_text = inp_lat_arrow_text.copy().set_color_by_gradient(WHITE, BLUE)
    white_out_lat_arrow_text = out_lat_arrow_text.copy().set_color_by_gradient(WHITE, BLUE)
    white_inp_lat_mesh_v = deepcopy(inp_lat_mesh_v)
    white_inp_lat_mesh_e = deepcopy(inp_lat_mesh_e)
    self.remove(*inp_points, *out_points, inp_square, out_square, *inp_lat_points,
                inp_square_text, out_square_text,inp_lat_arrow,out_lat_arrow,
               inp_lat_arrow_text,out_lat_arrow_text,*inp_lat_mesh_v,*inp_lat_mesh_v)
    self.add(white_inp_square, white_out_square, white_inp_square_text, white_out_square_text,
            white_inp_lat_arrow,white_out_lat_arrow,white_inp_lat_arrow_text,white_out_lat_arrow_text,
            *white_inp_lat_mesh_v, *white_inp_lat_mesh_e)
    triangle = RegularPolygon(n=3).move_to(white_inp_square.get_center()).set_color(YELLOW).set_fill(opacity=1).scale(0.1)
    t_a,t_b,t_c = triangle.get_vertices()[:3]
    t_ab = Line(t_a, t_b+1.5*(t_b-t_a))
    t_ac = Line(t_a, t_c+1.5*(t_c-t_a))
    #p_a = Dot(t_a, color=RED) ; p_b = Dot(t_b, color=BLUE) ; p_c = Dot(t_c, color=GREEN)
    inp_view = VGroup(triangle, t_ac, t_ab)#, p_a, p_b, p_c)
    lat_view = deepcopy(inp_view).shift(inp_lat_square.get_center()-white_inp_square.get_center())
    self.play(
      FadeIn(inp_view),
      FadeIn(lat_view),
      run_time=1*self.time_factor)
    self.wait(1*self.time_factor)
    self.play(
      ApplyMethod(inp_view.rotate, np.pi/2.),
      ApplyMethod(lat_view.rotate, np.pi/2.),
      run_time=1*self.time_factor)
    bed = Bed().scale(0.2).next_to(inp_view, RIGHT)
    self.play(FadeIn(bed), run_time=1*self.time_factor)
    lat_bed = Square(side_length=0.05, color=SUPER_BLUE).move_to(
      bed.get_center()+inp_lat_square.get_center()-white_inp_square.get_center())
    bed_arrow = Line(bed.get_center(), lat_bed.get_center()).add_tip().set_color([WHITE,BLUE]).set_sheen_direction(RIGHT)
    self.play(GrowArrow(bed_arrow),
              run_time=1*self.time_factor)
    self.play(
      FadeOut(bed_arrow),
      FadeIn(lat_bed),
      run_time=1*self.time_factor)
    self.representation_function_process(inp_lat_square, white_inp_lat_mesh_v,
                                         white_inp_lat_mesh_e, [lat_bed])
    self.play(
      ApplyMethod(inp_view.rotate, np.pi/2.),
      ApplyMethod(lat_view.rotate, np.pi/2.),
      run_time=1*self.time_factor)
    self.play(
      ApplyMethod(inp_view.shift, UP),
      ApplyMethod(lat_view.shift, UP),
      run_time=1*self.time_factor)
    self.play(
      ApplyMethod(inp_view.rotate, np.pi/2.),
      ApplyMethod(lat_view.rotate, np.pi/2.),
      run_time=1*self.time_factor)
    cabinet = Cabinet().scale(0.2).next_to(inp_view, LEFT)
    self.play(FadeIn(cabinet), run_time=1*self.time_factor)
    lat_cabinet = Square(side_length=0.05, color=WHITE).move_to(
      cabinet.get_center()+inp_lat_square.get_center()-white_inp_square.get_center())
    cabinet_arrow = Line(cabinet.get_center(), lat_cabinet.get_center()).add_tip().set_color([WHITE,BLUE]).set_sheen_direction(RIGHT)
    self.play(GrowArrow(cabinet_arrow),
              run_time=1*self.time_factor)
    self.play(
      FadeOut(cabinet_arrow),
      FadeIn(lat_cabinet),
      run_time=1*self.time_factor)
    self.representation_function_process(inp_lat_square, white_inp_lat_mesh_v,
                                         white_inp_lat_mesh_e, [lat_cabinet])
    self.play(
      ApplyMethod(inp_view.rotate, -np.pi/2.),
      ApplyMethod(lat_view.rotate, -np.pi/2.),
      run_time=1*self.time_factor)
    self.play(
      ApplyMethod(inp_view.shift, UP),
      ApplyMethod(lat_view.shift, UP),
      run_time=1*self.time_factor)
    fridge = Fridge().scale(0.25).next_to(inp_view, UP)
    self.play(FadeIn(fridge), run_time=1*self.time_factor)
    lat_fridge = Square(side_length=0.05, color=average_color(BLUE, WHITE)).move_to(
      fridge.get_center()+inp_lat_square.get_center()-white_inp_square.get_center())
    fridge_arrow = Line(fridge.get_center(), lat_fridge.get_center()).add_tip().set_color([WHITE,BLUE]).set_sheen_direction(RIGHT)
    fridge_arrow.set_sheen_direction(RIGHT)
    up_inp_lat_square = inp_lat_square.copy().shift(2*UP)
    up_inp_lat_mesh_v, up_inp_lat_mesh_e = mesh_square(up_inp_lat_square, 4, color=up_inp_lat_square.get_color(),
                                                 fill_color=up_inp_lat_square.get_color())
    up_inp_lat_mesh_v[0].set_color(WHITE)
    up_inp_lat_mesh_v[4].set_color(WHITE)
    #up_inp_lat_square.set_color(GREEN)
    #dot = Dot(up_inp_lat_square.get_center(), color=RED)
    self.play(
      #FadeIn(up_inp_lat_square),
      #FadeIn(dot),
      *[FadeIn(_) for _ in [up_inp_lat_square]+up_inp_lat_mesh_v+up_inp_lat_mesh_e],
      run_time=2*self.time_factor)
    self.play(GrowArrow(fridge_arrow),
              run_time=1*self.time_factor)
    self.play(
      FadeOut(fridge_arrow),
      FadeIn(lat_fridge),
      run_time=1*self.time_factor)
    self.representation_function_process(up_inp_lat_square, up_inp_lat_mesh_v,
                                         up_inp_lat_mesh_e, [lat_fridge])

class OpeningManimExample(Scene):
    def construct(self):
        title = TextMobject("This is some \\LaTeX")
        basel = TexMobject(
            "\\sum_{n=1}^\\infty "
            "\\frac{1}{n^2} = \\frac{\\pi^2}{6}"
        )
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeInFrom(basel, UP),
        )
        self.wait()

        transform_title = TextMobject("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*map(FadeOutAndShiftDown, basel)),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = TextMobject("This is a grid")
        grid_title.scale(1.5)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeInFromDown(grid_title),
            ShowCreation(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = TextMobject(
            "That was a non-linear function \\\\"
            "applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.apply_function,
            lambda p: p + np.array([
                np.sin(p[1]),
                np.sin(p[0]),
                0,
            ]),
            run_time=3,
        )
        self.wait()
        self.play(
            Transform(grid_title, grid_transform_title)
        )
        self.wait()

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(ShowCreation(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))


class WarpSquare(Scene):
    def construct(self):
        square = Square()
        self.play(ApplyPointwiseFunction(
            lambda point: complex_to_R3(np.exp(R3_to_complex(point))),
            square
        ))
        self.wait()


class WriteStuff(Scene):
    def construct(self):
        example_text = TextMobject(
            "This is a some text",
            tex_to_color_map={"text": YELLOW}
        )
        example_tex = TexMobject(
            "\\sum_{k=1}^\\infty {1 \\over k^2} = {\\pi^2 \\over 6}",
        )
        group = VGroup(example_text, example_tex)
        group.arrange(DOWN)
        group.set_width(FRAME_WIDTH - 2 * LARGE_BUFF)

        self.play(Write(example_text))
        self.play(Write(example_tex))
        self.wait()


class UdatersExample(Scene):
    def construct(self):
        decimal = DecimalNumber(
            0,
            show_ellipsis=True,
            num_decimal_places=3,
            include_sign=True,
        )
        square = Square().to_edge(UP)

        decimal.add_updater(lambda d: d.next_to(square, RIGHT))
        decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
        self.add(square, decimal)
        self.play(
            square.to_edge, DOWN,
            rate_func=there_and_back,
            run_time=5,
        )
        self.wait()

# See old_projects folder for many, many more
