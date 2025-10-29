"""
CS 589 - Assignment 2: ElasticSearch Ranking Evaluation
Complete implementation combining all functionality:
- Data conversion to JSON batches
- Bulk indexing automation
- Ratings generation (Algorithm 2)
- Ranking evaluation (Algorithm 1)

Student: Nakul Jadeja
"""

import json
import os
import subprocess
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
from elasticsearch import Elasticsearch

# Initialize Elasticsearch
es = Elasticsearch()

# ============================================================================
# PART 1: DATA CONVERSION
# ============================================================================

def convert_to_bulk_json_batches(input_file, output_prefix, batch_size=10000):
    """
    Convert qid2all.txt files to ElasticSearch bulk indexing JSON format
    
    Args:
        input_file: path to input file (e.g., 'python_qid2all.txt')
        output_prefix: prefix for output files (e.g., 'python')
        batch_size: documents per batch file (default 10000)
    """
    print(f"\n{'='*60}")
    print(f"Converting {input_file} to batch JSON files...")
    print(f"{'='*60}")
    
    batch_num = 0
    line_count = 0
    total_docs = 0
    
    outfile = open(f'{output_prefix}_batch_{batch_num}.json', 'w', encoding='utf-8')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                parts = line.strip().split('\t')
                
                if len(parts) == 4:
                    qid, title, question, answer = parts
                    
                    # Action line
                    action = {"index": {"_id": qid}}
                    outfile.write(json.dumps(action) + '\n')
                    
                    # Document data
                    doc = {
                        "title": title,
                        "body": question,
                        "answer": answer
                    }
                    outfile.write(json.dumps(doc) + '\n')
                    
                    line_count += 1
                    total_docs += 1
                    
                    if line_count >= batch_size:
                        outfile.close()
                        print(f"✓ Created {output_prefix}_batch_{batch_num}.json ({line_count} documents)")
                        batch_num += 1
                        line_count = 0
                        outfile = open(f'{output_prefix}_batch_{batch_num}.json', 'w', encoding='utf-8')
        
        outfile.close()
        if line_count > 0:
            print(f"✓ Created {output_prefix}_batch_{batch_num}.json ({line_count} documents)")
        
        print(f"\nTotal: {total_docs} documents in {batch_num + 1} batch files\n")
        return batch_num + 1
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        outfile.close()
        return 0


def convert_all_datasets():
    """
    Convert all three datasets to JSON batches
    """
    print("\n" + "="*60)
    print("STEP 1: CONVERTING ALL DATASETS TO JSON")
    print("="*60)
    
    datasets = ['python', 'java', 'javascript']
    batch_counts = {}
    
    for dataset in datasets:
        input_file = f'{dataset}_qid2all.txt'
        
        if not os.path.exists(input_file):
            print(f"ERROR: File not found: {input_file}")
            continue
        
        num_batches = convert_to_bulk_json_batches(input_file, dataset, batch_size=10000)
        batch_counts[dataset] = num_batches
    
    print("="*60)
    print("✓ ALL DATASETS CONVERTED!")
    print("="*60)
    return batch_counts


# ============================================================================
# PART 2: BULK INDEXING
# ============================================================================

def bulk_index_batches(lang, index_name, num_batches):
    """
    Bulk index all batch files for a specific language and index
    
    Args:
        lang: 'python', 'java', or 'javascript'
        index_name: name of ElasticSearch index
        num_batches: number of batch files to index
    """
    print(f"\n{'='*60}")
    print(f"Bulk indexing {lang} into {index_name}")
    print(f"{'='*60}")
    
    for i in range(num_batches):
        batch_file = f"{lang}_batch_{i}.json"
        
        if not os.path.exists(batch_file):
            print(f"WARNING: Batch file not found: {batch_file}")
            continue
        
        cmd = [
            'curl.exe', '-s', '-H', 'Content-Type: application/json',
            '-XPOST', f'localhost:9200/{index_name}/_doc/_bulk',
            '--data-binary', f'@{batch_file}'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Indexed {batch_file} into {index_name}")
            else:
                print(f"✗ Failed to index {batch_file}: {result.stderr}")
        except Exception as e:
            print(f"ERROR indexing {batch_file}: {str(e)}")


def bulk_index_all(batch_counts):
    """
    Bulk index all datasets into all indices
    """
    print("\n" + "="*60)
    print("STEP 2: BULK INDEXING ALL DATA")
    print("="*60)
    
    similarities = ['bm25', 'tfidf', 'lmd']
    
    for lang, num_batches in batch_counts.items():
        for sim in similarities:
            index_name = f"{lang}_{sim}"
            bulk_index_batches(lang, index_name, num_batches)
    
    print("\n" + "="*60)
    print("✓ ALL BULK INDEXING COMPLETE!")
    print("="*60)


# ============================================================================
# PART 3: GENERATE RATINGS (Algorithm 2)
# ============================================================================

def generate_ratings(lang, index_name):
    """
    Algorithm 2: Generate ratings for each query
    
    Args:
        lang: 'python', 'java', or 'javascript'
        index_name: name of the index
    
    Returns:
        Dictionary mapping qid1 -> ratings list
    """
    cosidf_file = f"{lang}_cosidf.txt"
    
    if not os.path.exists(cosidf_file):
        print(f"ERROR: File not found: {cosidf_file}")
        return None
    
    print(f"Reading {cosidf_file}...")
    
    # Read cosidf file
    qid_dataframe = pd.read_csv(
        cosidf_file,
        sep="\t",
        usecols=["qid1", "qid2", "label"],
        dtype={"qid1": str, "qid2": str, "label": int}
    )
    
    all_ratings = {}
    unique_qid1s = qid_dataframe['qid1'].unique()
    print(f"Found {len(unique_qid1s)} unique qid1s")
    
    # For each qid1
    for qid1 in unique_qid1s:
        ratings = []
        query_data = qid_dataframe[qid_dataframe['qid1'] == qid1]
        
        # Algorithm 2, lines 3-5
        for _, row in query_data.iterrows():
            ratings.append({
                "_index": index_name,
                "_id": row['qid2'],
                "rating": int(row['label'])
            })
        
        all_ratings[qid1] = ratings
    
    # Algorithm 2, line 6: Save ratings as JSON
    output_file = f"{lang}_{index_name}_ratings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_ratings, f, indent=2)
    
    print(f"✓ Saved ratings to {output_file}\n")
    return all_ratings


def generate_all_ratings():
    """
    Generate ratings for all 9 combinations
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING RATINGS (Algorithm 2)")
    print("="*60)
    
    languages = ['python', 'java', 'javascript']
    similarities = ['bm25', 'tfidf', 'lmd']
    
    for lang in languages:
        for sim in similarities:
            index_name = f"{lang}_{sim}"
            print(f"\nGenerating ratings for {index_name}...")
            generate_ratings(lang, index_name)
    
    print("="*60)
    print("✓ ALL RATINGS GENERATED!")
    print("="*60)


# ============================================================================
# PART 4: RANKING EVALUATION (Algorithm 1)
# ============================================================================

def ranking(qid1, qid1_title, ratings):
    """
    Create ranking query structure (Figure 2 from assignment)
    """
    _search = {
        "requests": [
            {
                "id": str(qid1),
                "request": {
                    "query": {
                        "bool": {
                            "must_not": {
                                "match": {"_id": qid1}
                            },
                            "should": [
                                {
                                    "match": {
                                        "title": {
                                            "query": qid1_title,
                                            "boost": 3.0,
                                            "analyzer": "my_analyzer"
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "body": {
                                            "query": qid1_title,
                                            "boost": 0.5,
                                            "analyzer": "my_analyzer"
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "answer": {
                                            "query": qid1_title,
                                            "boost": 0.5,
                                            "analyzer": "my_analyzer"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
                "ratings": ratings
            }
        ],
        "metric": {
            "dcg": {
                "k": 10,
                "normalize": True
            }
        }
    }
    return _search


def evaluate_index(lang, similarity, index_name, ratings_file):
    """
    Algorithm 1: Ranking Function
    Evaluates NDCG@10 for a specific index
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {lang.upper()} - {similarity.upper()}")
    print(f"Index: {index_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(ratings_file):
        print(f"ERROR: Ratings file not found: {ratings_file}")
        return 0.0
    
    # Load ratings
    with open(ratings_file, 'r', encoding='utf-8') as f:
        all_ratings_dict = json.load(f)
    
    # Algorithm 1, line 3
    ndcg_list = []
    processed = 0
    errors = 0
    
    # Algorithm 1, line 4: For each qid1
    for qid1, ratings in all_ratings_dict.items():
        try:
            # Algorithm 1, line 5
            response = es.get(index=index_name, doc_type='_doc', id=qid1)
            qid1_title = response['_source']['title']
            
            # Algorithm 1, line 7
            _search = ranking(qid1, qid1_title, ratings)
            
            # Algorithm 1, line 8
            result = es.rank_eval(index=index_name, body=_search)
            
            # Algorithm 1, line 9
            ndcg = result['metric_score']
            
            # Algorithm 1, line 10
            ndcg_list.append(ndcg)
            processed += 1
            
            if processed % 100 == 0:
                current_avg = sum(ndcg_list) / len(ndcg_list)
                print(f"Progress: {processed} queries | Avg: {current_avg:.4f}")
                
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Error qid1={qid1}: {str(e)}")
    
    # Algorithm 1, line 11
    if ndcg_list:
        avg_ndcg = sum(ndcg_list) / len(ndcg_list)
        print(f"\nProcessed: {processed} | Errors: {errors}")
        print(f"Average NDCG@10: {avg_ndcg:.4f}\n")
        return avg_ndcg
    else:
        print("ERROR: No valid scores!")
        return 0.0


def evaluate_all():
    """
    Evaluate all 9 combinations
    """
    print("\n" + "="*60)
    print("STEP 4: RANKING EVALUATION (Algorithm 1)")
    print("="*60)
    
    languages = ['python', 'java', 'javascript']
    similarities = {
        'bm25': 'BM25',
        'tfidf': 'TF-IDF',
        'lmd': 'Dirichlet LM'
    }
    
    results = {}
    
    for lang in languages:
        results[lang] = {}
        for sim_code, sim_name in similarities.items():
            index_name = f"{lang}_{sim_code}"
            ratings_file = f"{lang}_{index_name}_ratings.json"
            
            ndcg_score = evaluate_index(lang, sim_name, index_name, ratings_file)
            results[lang][sim_code] = ndcg_score
    
    # Print final table
    print("\n" + "="*80)
    print("FINAL RESULTS - NDCG@10 SCORES")
    print("="*80)
    print(f"{'Dataset':<15} {'BM25':<20} {'TF-IDF':<20} {'Dirichlet LM':<20}")
    print("-"*80)
    
    for lang in languages:
        print(f"{lang.capitalize():<15} "
              f"{results[lang]['bm25']:.4f}{'':>15} "
              f"{results[lang]['tfidf']:.4f}{'':>15} "
              f"{results[lang]['lmd']:.4f}{'':>15}")
    
    print("="*80)
    
    # Save results
    with open('ndcg_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('ndcg_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("NDCG@10 EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Dataset':<15} {'BM25':<20} {'TF-IDF':<20} {'Dirichlet LM':<20}\n")
        f.write("-"*80 + "\n")
        for lang in languages:
            f.write(f"{lang.capitalize():<15} "
                   f"{results[lang]['bm25']:.4f}{'':>15} "
                   f"{results[lang]['tfidf']:.4f}{'':>15} "
                   f"{results[lang]['lmd']:.4f}{'':>15}\n")
        f.write("="*80 + "\n")
    
    print("\n✓ Results saved to ndcg_results.json and ndcg_report.txt")
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run complete assignment pipeline
    """
    print("\n" + "="*80)
    print(" "*20 + "CS 589 - ASSIGNMENT 2")
    print(" "*15 + "ElasticSearch Ranking Evaluation")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print("="*80)
    
    # Step 1: Convert data to JSON batches
    batch_counts = convert_all_datasets()
    
    # Step 2: Bulk index all data
    # NOTE: This requires curl command availability
    # You may need to run bulk indexing manually if curl is not available
    response = input("\nProceed with bulk indexing? This may take a while. (y/n): ")
    if response.lower() == 'y':
        bulk_index_all(batch_counts)
    else:
        print("Skipping bulk indexing. Please index manually.")
    
    # Step 3: Generate ratings
    generate_all_ratings()
    
    # Step 4: Evaluate rankings
    results = evaluate_all()
    
    print("\n" + "="*80)
    print("✓ ASSIGNMENT COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nSubmission files created:")
    print("  - ndcg_results.json")
    print("  - ndcg_report.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()