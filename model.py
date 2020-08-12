import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import math
import torch
from resnet import ResNetEncoder

class Policy(nn.Module):
    def __init__(self, env_config, num_players=2, max_entities=10):
        super(Policy, self).__init__()
        # State Parser
        self.parse_state = ParseState(env_config, 10)

        self.MAX_ENTITIES = 10
        # Map Encoder
        self.map = MapEmbedding(128)
        # Entity Encoder
        self.entity = EntityEmbedding(128, env_config['size'], 1)
        # Scalar Encoder
        self.scalar_encoder = nn.Linear(num_players, 128)
        # transformer

        # self.max_entities = 10

        self.action_map = [None, "NORTH", "EAST", "SOUTH", "WEST", "CONVERT", "SPAWN"]

        self.SHIP_TYPE = 2
        self.SHIPYARD_TYPE = 1

        num_actions = len(self.action_map)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=100), 2, norm=None)
        self.fc = nn.Linear(128, num_actions)

        self.softmax = nn.Softmax(-1)
   
    '''
    {
        halite
        obstruction
        entities
    }
    '''

    def device(self):
        return next(self.parameters()).device

        
    def forward(self, state):
        # Scalar encoding
        scalar = state['scalar'].to(self.device())
        scalar_encoding = F.relu(self.scalar_encoder(scalar)).unsqueeze(1)
    
        # Spatial Encoding
        game_map = state['map'].to(self.device())
        map_encoding = self.map(game_map).unsqueeze(1)
        
        # Entity Encoding
        entity_typ = state['entity_typ'].to(self.device())
        entity_pos = state['entity_pos'].to(self.device())
        entity_scalar = state['entity_scalar'].to(self.device())

        entity_encodings = self.entity(entity_typ, entity_pos, entity_scalar)

        embeddings = map_encoding + entity_encodings + scalar_encoding

        set_embedding =  self.transformer(embeddings)
        
        out = self.fc(set_embedding)
       
        return self.softmax(out)

    def action(self, state):
        t_state = self.parse_state(state)
        out = self.forward(t_state)

        raw_actions = Categorical(probs=out[0]).sample()
        # raw_actions = torch.argmax(out, -1)[0]
        actions = {}

        n_entities = len(t_state['entity_id'])

        mask = torch.tensor([1] * n_entities + [0] * (self.MAX_ENTITIES - n_entities)).to(self.device())

        
        # TODO: Migrate this code to env helper
        for e, eid in enumerate(t_state['entity_id']):
            act = self.action_map[raw_actions[e]]

            typ = t_state['entity_typ'][0][e]
            if typ == self.SHIP_TYPE and act == "SPAWN":
                act = None
            elif typ == self.SHIPYARD_TYPE and (act != "SPAWN" and act != None):
                act = None
            elif typ == 0:
                continue

            if act == "SPAWN":
                if n_entities < self.MAX_ENTITIES:
                    n_entities += 1
                else:
                    act = None
            
            if act is not None:
                actions[eid] = act
   
        return actions, out, raw_actions, mask


class ParseState(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, config, max_entities):

        self.map_size = config['size']
        self.max_halite = config['maxCellHalite']
        self.starting_halite = config['startingHalite']
        self.max_entities = max_entities

    def __call__(self, s):
        step = s['step']

        halite = torch.tensor(s['halite']).float()
        halite = halite.reshape(self.map_size, self.map_size, 1) / self.max_halite
        obstruction = torch.zeros(self.map_size**2).float()

        me = s['players'][0]
        my_halite, my_shipyards, my_ships = tuple(me)
        
        scalar = torch.zeros(len(s['players']))

        scalar[0] = my_halite


        entity_typ = []
        entity_pos = []
        entity_scalar = []
        entity_id = []

        for shipyard_id, shipyard_pos in my_shipyards.items():
            obstruction[shipyard_pos] = 1.0
            x = int(shipyard_pos % self.map_size)
            y = int(shipyard_pos / self.map_size)
            entity_typ.append(1)
            entity_pos.append([x,y])
            entity_scalar.append([0])
            entity_id.append(shipyard_id)


        for ship_id, ship_pos in my_ships.items():
            obstruction[ship_pos[0]] = 1.0
            x = int(ship_pos[0] % self.map_size)
            y = int(ship_pos[0] / self.map_size)
            entity_typ.append(2)
            entity_pos.append([x,y])
            entity_scalar.append([ship_pos[1]])
            entity_id.append(ship_id)


        opponents = s['players'][1:]

        for i, opponent in enumerate(opponents):
            opp_halite, opp_shipyards, opp_ships = tuple(opponent)
            scalar[i+1] = opp_halite
            for shipyard_pos in opp_shipyards.values():
                obstruction[shipyard_pos] = 1.0
            for ship_pos in opp_ships.values():
                obstruction[ship_pos[0]] = 1.0

        obstruction = obstruction.reshape(self.map_size, self.map_size, 1)
    
        spat_map = torch.cat((halite, obstruction), 2).unsqueeze(0).permute(0,3,1,2)

        n_entities = len(entity_id)
        diff = self.max_entities - n_entities

        entity_typ = F.pad(torch.tensor(entity_typ).long().unsqueeze(0), (0, diff), "constant", 0)
        entity_pos =  F.pad(torch.tensor(entity_pos).long().unsqueeze(0), (0, 0, 0, diff), "constant", 0)
        entity_scalar =  F.pad(torch.tensor(entity_scalar).float().unsqueeze(0), (0, 0, 0, diff), "constant", 0)
        
        scalar = scalar.unsqueeze(0) / self.starting_halite
        return {
            'map': spat_map,
            'entity_typ': entity_typ,
            'entity_pos': entity_pos,
            'entity_scalar': entity_scalar,
            'entity_id': entity_id,
            'scalar': scalar
        }
 

class MapEmbedding(nn.Module):
    def __init__(self, embed_size=256, depth=2, maps=2):
        super(MapEmbedding, self).__init__()

        blocks = []
        c_b = 64
        while c_b < embed_size:
            blocks.append(c_b)
            c_b *= 2
        blocks.append(embed_size)
        deepths = [depth] * len(blocks)
        self.resnet = ResNetEncoder(in_channels=maps, blocks_sizes=blocks, deepths=deepths)

    def forward(self, multi_layer_map):
        return self.resnet(multi_layer_map)
        


class EntityEmbedding(nn.Module):
    def __init__(self, d_model, map_size, n_scalars):
        super(EntityEmbedding, self).__init__()
        # self.lut = pre_trained.embeddings.word_embeddings
        self.EntityType = nn.Embedding(2 + 1, d_model)
        self.EntityPosition = PositionalEncoding2D(d_model, map_size, map_size)
        self.fc = nn.Linear(n_scalars, d_model)
        self.EntityType.weight.data.uniform_(-0.1, .1)

    def forward(self, typ, pos, scalar):
        return self.EntityType(typ) + self.EntityPosition(pos) + F.relu(self.fc(scalar))


# Retrieved from pytorch website

class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()

        if d_model % 4 != 0:
            raise Error()

        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pos_w = torch.arange(0., width).unsqueeze(1)

        pos_h = torch.arange(0., height).unsqueeze(1)

        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        # (*, 2)
        pos = pos.transpose(0, -1)
        return self.pe[:, pos[0], pos[1]].transpose(0, -1)
        

if __name__ == "__main__":

    pe = PositionalEncoding2D(8, 10, 10)

    pos = torch.tensor([[[0,0], [0,0], [9,9]],[[0,0], [0,0], [9,9]]])
    print(pe(pos))
