import torch
from torch import nn
from torch.autograd import Variable 
from torch.functional import F
from GEN.composer import GEN_Composer
from Baseline.composer import Baseline_Composer


class Classify(nn.Module):
    def __init__(self, baseline=False):
        super(Classify, self).__init__()

        if not baseline: 
            self.number_of_coordinates_copies = 8
            self.composer = GEN_Composer(num_copies=self.number_of_coordinates_copies)
            self.post_embedding_processor = PostEmbeddingProcessorGEN(num_copies=self.number_of_coordinates_copies)
        else: 
            self.number_of_coordinates_copies = 32
            self.composer = Baseline_Composer(num_copies=self.number_of_coordinates_copies)
            self.post_embedding_processor = PostEmbeddingProcessorBaseline(num_copies=self.number_of_coordinates_copies)
        
        # Learnable scalar
        self.scalar = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.candidates_encoder = CandidatesEncoder()  
        return

    def forward(self, x, v, v_q, candidates):
        # Obtain a bs x node_state_dim embedding for each query pose
        embedded_query_frames = self.composer(x, v, v_q)
        embedded_query_frames = embedded_query_frames.view(-1, embedded_query_frames.shape[2])

        # Feed to feed forward layers
        embedded_query_frames = self.post_embedding_processor(embedded_query_frames)

        # Convolutional encoding of candidates without any composer
        candidates_embeddings = self.candidates_encoder(candidates)
        return embedded_query_frames, candidates_embeddings


class CandidatesEncoder(nn.Module):
    def __init__(self):
        super(CandidatesEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv1bn = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv2bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv4bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7bn = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 254, kernel_size=1, stride=1)
        self.conv8bn = nn.BatchNorm2d(254)
        self.pool  = nn.AvgPool2d(16)

    def forward(self, x):
        skip_in  = F.relu(self.conv1bn(self.conv1(x)))
        skip_out = F.relu(self.conv2bn(self.conv2(skip_in)))

        r = F.relu(self.conv3bn(self.conv3(skip_in)))
        r = F.relu(self.conv4bn(self.conv4(r))) + skip_out
        
        skip_out  = F.relu(self.conv5bn(self.conv5(r)))

        r = F.relu(self.conv6bn(self.conv6(r)))
        r = F.relu(self.conv7bn(self.conv7(r))) + skip_out
        r = F.relu(self.conv8bn(self.conv8(r)))
        r = self.pool(r).squeeze(3).squeeze(2)
        return r


class PostEmbeddingProcessorGEN(nn.Module):
  def __init__(self, num_copies=None):
    super(PostEmbeddingProcessorGEN, self).__init__()
    in_sz = 256 + (num_copies*7)
    self.fc0 = nn.Linear(in_features=in_sz, out_features=256)
    self.fc0bn = nn.BatchNorm1d(256)
    self.fc1 = nn.Linear(in_features=256, out_features=254)
    self.fc1bn = nn.BatchNorm1d(254)

  def forward(self, x):
    x = F.relu(self.fc0bn(self.fc0(x)))
    return self.fc1bn(self.fc1(x))


class PostEmbeddingProcessorBaseline(nn.Module):
  def __init__(self, num_copies=None):
    super(PostEmbeddingProcessorBaseline, self).__init__()
    in_sz = 256 + (num_copies*7)
    self.fc0 = nn.Linear(in_features=in_sz, out_features=512)
    self.fc0bn = nn.BatchNorm1d(512)
    self.fc1 = nn.Linear(in_features=512, out_features=512)
    self.fc1bn = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(in_features=512, out_features=512)
    self.fc2bn = nn.BatchNorm1d(512)
    self.fc3 = nn.Linear(in_features=512, out_features=254)
    self.fc3bn = nn.BatchNorm1d(254)

  def forward(self, x):
    x = F.relu(self.fc0bn(self.fc0(x)))
    x = F.relu(self.fc1bn(self.fc1(x)))
    x = F.relu(self.fc2bn(self.fc2(x)))
    return self.fc3bn(self.fc3(x))