import torch
import wandb
#from pykeen.evaluation import RankBasedEvaluator
from types import MethodType
#from pykeen.utils import flatten_dictionary

def init_pykeen_metrics():
    metrics = ['AdjustedArithmeticMeanRank']
    metrics_kwargs = [None] * len(metrics)
    hits_at_k_keys = ['HitsAtK']
    ks = (1, 3, 5, 10)
    #call hits_at_k for each k in KS
    for hits_at_k_key in hits_at_k_keys:
        metrics += [hits_at_k_key] * len(ks)
        metrics_kwargs += [dict(k=k) for k in ks]
    evaluator = RankBasedEvaluator(filtered=True, metrics = metrics, metrics_kwargs = metrics_kwargs, add_defaults=False) #type: ignore
    return evaluator

#pykeen
@torch.no_grad()
def evaluate_pykeen(evaluator, pred,  obj, label, epoch, split, target = 'tail'):
    """
        evaluate the model on by metrics in pykeen

    :param pred: a 2D bs, ne tensor containing bs distributions over entities [batch, n_target]
    :param obj: the actual objects being predicted [batch] (pred ent's index)
    :param label: a 2D bs, ne multi-hot tensor [batch, n_target] (true label is 1)
        (where 1 -> the obj appeared in train/val/test split)
    output: evaluator aftaer processing scores
    """

    # filter out irrelevant true entities, set their socres to  -1000000
    b_range = torch.arange(pred.size()[0], device=pred.device)
    if split != 'train':
        irrelevant = label.clone() #true target is 1, others are nearly 0
        irrelevant[b_range, obj] = 0
        pred[irrelevant.bool()] = float("nan") #irrelevant true target is 1, set them to Nan

    true_scores = pred[b_range, obj]

    #input all_scores and true scores
    evaluator.process_scores_(
        hrt_batch = None,
        target = target,
        true_scores=true_scores[:, None], #[batch,1]
        scores = pred #[batch, target]
    ) #store true ranks in evaluator

    return evaluator


@torch.no_grad()
def average_pykeen_metrics( evaluator, epoch, split, use_wandb):
    def to_my_dict(self):
        return {f"{side}/{metric_name}": value for side, rank_type, metric_name, value in self._iter_rows() if rank_type == 'realistic'}
    #calculate average pykeen metrics
    # Finalize(average metrics of stored metric list in evaluator)
    metric_results = evaluator.finalize()
    metric_results.to_my_dict = MethodType(to_my_dict, metric_results)

    # add all metrics to wandb, including both/head/tail and optimistic/pessimistic
    metrics = flatten_dictionary(dictionary=metric_results.to_my_dict(), prefix=split)
    #if use_wandb:
        #for k, v in metrics.items():
            #wandb.log({k: v, "epoch": epoch})
    return metrics


@torch.no_grad()
def compute( pred, obj, label, results, split = 'test'):
    """
        Discard the predictions for all objects not in label (not currently evaluated)

    :param pred: a 2D bs, ne tensor containing bs distributions over entities [batch, n_target]
    :param obj: the actual objects being predicted [batch] (pred ent's index)
    :param label: a 2D bs, ne multi-hot tensor [batch, n_target] (true label is 1)
        (where 1 -> the obj appeared in train/val/test split)
    :param ignored_entities: some entities we expect to not appear in s/o positions.
        can mention them here. Its a list like [2, 10, 3242344, ..., 69]
    :param results:
    :return: add each sample's metrics to results
    """
    #ignored_entities = self.excluding_entities  # remove qualifier only entities if the flag says so
    b_range = torch.arange(pred.size()[0], device=pred.device)
    if split != 'train':
        irrelevant = label.clone()
        irrelevant[b_range, obj] = 0  #irrelevant true target is 1
        #irrelevant[:, ignored_entities] = 1  # Across batch, add 1 to ents never to be predicted
        pred[irrelevant.bool()] = -1000000 #chage other true target to -1000

    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
        b_range, obj]
    # results = {}
    ranks = ranks.float()
    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
    results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
    results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
    #results['log-mrr'] = torch.log(torch.sum(1.0 / ranks)+1).item() + results.get('log-mrr', 0.0)
    #results['0.5-mrr'] = torch.pow(torch.sum(1.0 / ranks), 0.5).item() + results.get('0.5-mrr', 0.0)

    for k in [0, 4, 9]:
        results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
            'hits@{}'.format(k + 1), 0.0)
    return results

def summarize_metrics(accumulated_metrics: dict, eval_size: int) -> dict:
    """
        Aggregate metrics across time. Accepts np array of (len(self.data_eval), len(self.metrics))
    """
    # mean = np.mean(accumulated_metrics, axis=0)
    summary = {}

    for k, v in accumulated_metrics.items():
        summary[k] = v / float(eval_size) if k != 'count' else v

    return summary