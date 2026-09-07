"""Combined CSV v1 for Python benchmark and framework runners; native CSV contains measurements only."""
FIELDS = ('schema_version,runner,backend,strategy,graph_path,features_path,model_path,model_types,feature_widths,'
          'nodes,stored_edges,layers,orientation,dtype,seed,provenance,threads,schedule,block_size,'
          'warmups,repetitions,verification,atol,rtol,sample,load_ms,setup_ms,upload_ms,reset_ms,compute_ms,'
          'download_ms,end_to_end_ms,mean_ms,stddev_ms,workspace_bytes,host_peak_bytes,device_peak_bytes,'
          'memory_policy,hardware,software,command,timing_policy').split(',')
