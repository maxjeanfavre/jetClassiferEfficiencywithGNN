import torch
import torch.nn as nn


class EdgeNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        outfeatures,
        dropout,
        output_name="edge_hidden_state",
        first=False,
    ):

        """
        Args:
            in_features : num of input features
        """

        super(EdgeNetwork, self).__init__()

        self.output_name = output_name
        self.first = first

        midfeatures = int((2 * in_features + outfeatures) / 2)
        self.net = nn.Sequential(
            nn.Linear(2 * in_features + 1, midfeatures),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(midfeatures, outfeatures),
            nn.Tanh(),
        )

    def forward(self, x):

        if self.first == True:
            input_data = torch.cat(
                [
                    x.dst["node_hidden_state"],
                    x.src["node_hidden_state"],
                    x.data["dR"].unsqueeze(1),
                ],
                dim=1,
            )

        else:
            input_data = torch.cat(
                [
                    x.dst["node_features"],
                    x.dst["node_hidden_state"],
                    x.src["node_features"],
                    x.src["node_hidden_state"],
                    x.data["dR"].unsqueeze(1),
                ],
                dim=1,
            )

        return {self.output_name: self.net(input_data)}


class NodeNetwork(nn.Module):
    def __init__(
        self, original_in_features, in_features, out_features, dropout, first=False
    ):
        super(NodeNetwork, self).__init__()

        self.first = first

        out_features = int(out_features / 2)
        mid_features = int((original_in_features + in_features + out_features) / 2)

        self.net1 = nn.Sequential(
            nn.Linear(original_in_features + in_features, mid_features, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(mid_features, out_features, bias=True),
            nn.Tanh(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(out_features, out_features, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):

        message_sum = torch.sum(x.mailbox["edge_hidden_state"], dim=1)

        if self.first == True:
            out1 = self.net1(x.data["node_hidden_state"])
        else:
            out1 = self.net1(
                torch.cat((x.data["node_features"], x.data["node_hidden_state"]), dim=1)
            )

        out2 = self.net2(message_sum)

        out = torch.cat([out1, out2], dim=1)
        out = out / torch.norm(out, p="fro", dim=1, keepdim=True)

        return {"node_hidden_state": out}


class JetEfficiencyNet(nn.Module):
    def __init__(
        self,
        in_features,
        feats,
        correction_layers,
        num_classes,
        flavour_embedding_num_embeddings,
        flavour_embedding_dim,
        edge_network_dropout,
        node_network_dropout,
        eff_correction_dropout,
    ):

        """
        Args:
            in_features        : number of input feature for each node
            feats              : list of features for the deepset
            correction_layers  : list of features in the hidden layers of the effeciency prediction network
        """

        super(JetEfficiencyNet, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=flavour_embedding_num_embeddings,
            embedding_dim=flavour_embedding_dim,
        )

        self.node_updates = nn.ModuleList()
        self.edge_updates = nn.ModuleList()

        edge_network = EdgeNetwork(
            in_features=in_features,
            outfeatures=int(feats[0] / 2),
            dropout=edge_network_dropout,
            first=True,
        )
        node_network = NodeNetwork(
            original_in_features=in_features,
            in_features=0,
            out_features=feats[0],
            dropout=node_network_dropout,
            first=True,
        )

        self.node_updates.append(node_network)
        self.edge_updates.append(edge_network)

        for i in range(1, len(feats)):
            edge_network = EdgeNetwork(
                in_features=in_features + feats[i - 1],
                outfeatures=int(feats[i] / 2),
                dropout=edge_network_dropout,
                first=False,
            )
            node_network = NodeNetwork(
                original_in_features=in_features,
                in_features=feats[i - 1],
                out_features=feats[i],
                dropout=node_network_dropout,
                first=False,
            )

            self.node_updates.append(node_network)
            self.edge_updates.append(edge_network)

        eff_correction_layers = []
        eff_correction_layers.append(
            nn.Linear(in_features + feats[-1], correction_layers[0])
        )
        eff_correction_layers.append(nn.ReLU())

        for hidden_i in range(1, len(correction_layers)):
            eff_correction_layers.append(
                nn.Linear(correction_layers[hidden_i - 1], correction_layers[hidden_i])
            )
            if eff_correction_dropout is not None:
                eff_correction_layers.append(nn.Dropout(p=eff_correction_dropout))
            eff_correction_layers.append(nn.ReLU())

        eff_correction_layers.append(nn.Linear(correction_layers[-1], num_classes))

        self.eff_correction = nn.Sequential(*eff_correction_layers)

    def forward(self, g):

        # pass flav index through embedding and cat it to node_features
        g.ndata["node_features"] = torch.cat(
            (g.ndata["node_features"], self.embedding(g.ndata["flav_indices"])), dim=1
        )

        # don't want init, so node_hidden_state = node_features to start with
        g.ndata["node_hidden_state"] = g.ndata["node_features"]

        for i in range(len(self.node_updates)):
            g.update_all(self.edge_updates[i], self.node_updates[i])

        cat = torch.cat((g.ndata["node_features"], g.ndata["node_hidden_state"]), dim=1)

        g.ndata["node_prediction"] = self.eff_correction(cat)
        out = g.ndata["node_prediction"]

        return out
