            # if confidence in test_batch_on_confidences:
                
            #     for batch_size in [2**i for i in range(12)]:
            #         batch_data_loader = DataLoader(test_dataset, 
            #                  batch_size, True, num_workers = 0)
            #         batch_total_time = 0
            #         batch_ttr = 0
            #         batch_nc_total_time = 0
            #         num_batch = 0
            #         total = 0
            #         batch_ttr = 0

            #         for images, labels in test_data_loader:
            #             num_batch+=1
            #             total+= labels.size(0)
            #             images = images.to(conf.test_device)
            #             model.backbone.config_cache(active=False, threshold = confidence)
            #             _, nc_results = model(images)
            #             batch_nc_time = nc_results["end_time"] - nc_results["start_time"]
            #             batch_nc_total_time += batch_nc_time
            #             model.backbone.config_cache(active=True, shrink=True, threshold = confidence)
            #             _, results = model(images)
            #             batch_start_time = results["start_time"]
            #             batch_end_time = results["end_time"]
            #             batch_tt = batch_end_time - batch_start_time
            #             batch_total_time += batch_tt
            #             for i in range(conf.num_exits + 1):
            #                 if len(results["hits"]) == i:
            #                     # print(f"All samples in the batch#{num_batch} have been resolved before exit#{i}")
            #                     break
            #                 hits = results["hits"][i]
            #                 num_hits = torch.sum(hits).item()
            #                 hit_time = (results["hit_times"][i] - batch_start_time)
            #                 batch_ttr += num_hits * hit_time
            #         batch_df = batch_df.append({
            #             "BatchSize": batch_size,
            #             "Confidence": confidence,
            #             "ResponseTime":round(batch_nc_total_time, 4),
            #             "CachedResponseTime": round(batch_total_time, 4),
            #             "MTTR": round(batch_nc_total_time/num_batch, 4),
            #             "CachedMTTR": round(batch_ttr/total, 4),
            #             "MTTRRatio": round(100 * (batch_ttr/total)/(batch_nc_total_time/num_batch), 2)
            #         }, ignore_index=True)
            #         print(batch_df)
            #         batch_df.to_csv(f"{conf.report_dir}/batch.csv", index_label="Idx")