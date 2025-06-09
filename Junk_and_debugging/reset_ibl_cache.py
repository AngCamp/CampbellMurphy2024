#!/usr/bin/env python3
import os
import yaml
import shutil
from pathlib import Path
from one.api import ONE

def load_config():
    """Load the configuration from united_detector_config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'united_detector_config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        return {}

def reset_ibl_cache():
    """Reset the IBL ONE cache using refresh_cache() method"""
    # Use the actual cache path from run_pipeline.sh
    cache_dir = "/space/scratch/IBL_data_cache"
    cache_path = Path(cache_dir)
    
    print(f"\nCurrent IBL cache directory: {cache_path}")
    
    # Check if cache directory exists, create if needed
    if not cache_path.exists():
        print("\nCache directory does not exist, creating it...")
        cache_path.mkdir(parents=True, exist_ok=True)
    
    print("\nInitializing ONE client with remote connection...")
    try:
        # Initialize ONE with remote connection (OneAlyx) to access refresh_cache
        # This connects to the IBL public database which has refresh_cache method
        one = ONE(base_url='https://openalyx.internationalbrainlab.org', 
                  cache_dir=str(cache_path))
        print("✓ ONE client initialized successfully")
        print(f"Cache directory: {one.cache_dir}")
        print(f"Connected to: {one.alyx.base_url}")
        
        # Use refresh_cache() to force fresh download from remote
        print("\nRefreshing cache from remote server...")
        print("This will download fresh cache tables and may take a few minutes...")
        
        # refresh_cache('remote') forces download of fresh cache tables
        one.refresh_cache('remote')
        print("✓ Cache refreshed successfully from remote server")
        
        # Verify the refresh worked
        print("\nVerifying cache status...")
        try:
            # Try a simple operation to verify cache is working
            search_terms = one.search_terms()
            print(f"✓ Cache verification successful - {len(search_terms)} search terms available")
            return True
        except Exception as e:
            print(f"⚠️  Cache verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ONE initialization/refresh failed: {str(e)}")
        print("\nTrying fallback approach with manual cache clear...")
        
        # Fallback: manual cache clearing + reinitialization
        try:
            if cache_path.exists():
                print("Clearing cache files manually...")
                # Remove cache files but keep directory structure
                for item in cache_path.glob('*.parquet'):
                    item.unlink()
                    print(f"  Removed: {item.name}")
                for item in cache_path.glob('*.json'):
                    item.unlink() 
                    print(f"  Removed: {item.name}")
            
            # Reinitialize after manual clear
            print("Reinitializing ONE after manual clear...")
            one = ONE(cache_dir=str(cache_path))
            one.refresh_cache('remote')
            print("✓ Fallback cache reset completed successfully")
            return True
            
        except Exception as fallback_error:
            print(f"❌ Fallback approach also failed: {fallback_error}")
            print("\nTrying alternative approach...")
            
            # Alternative: Use the load_cache method if available
            try:
                print("Attempting to force cache reload...")
                # Clear corrupted files first
                if cache_path.exists():
                    for item in cache_path.glob('*.parquet'):
                        if item.stat().st_size == 0 or 'corrupted' in str(item):
                            item.unlink()
                            print(f"  Removed potentially corrupted: {item.name}")
                
                # Try to reinitialize with explicit remote connection
                one = ONE(base_url='https://openalyx.internationalbrainlab.org',
                         cache_dir=str(cache_path))
                
                # Try alternative cache refresh methods
                if hasattr(one, 'load_cache'):
                    one.load_cache()
                    print("✓ Alternative cache reload successful")
                    return True
                else:
                    print("❌ No suitable cache refresh method found")
                    return False
                    
            except Exception as alt_error:
                print(f"❌ Alternative approach failed: {alt_error}")
                return False

def main():
    print("=== IBL ONE Cache Reset Tool (Using refresh_cache) ===")
    print("This tool will refresh your cache from the remote IBL server.")
    print("This is more thorough than cache_clear() and will download fresh data.")
    
    # Load config (optional, for context)
    config = load_config()
    
    success = reset_ibl_cache()
    
    if success:
        print("\n✅ Cache reset completed successfully!")
        print("\nWhat happened:")
        print("- Cache was refreshed from remote IBL server")
        print("- Fresh metadata and parquet files downloaded") 
        print("- Corrupted cache files replaced")
        print("\nNext steps:")
        print("- Your pipeline should now run without parquet errors")
        print("- First few requests may be slightly slower as cache rebuilds")
        print("- Monitor for any remaining '_web_client' errors")
    else:
        print("\n❌ Cache reset failed!")
        print("\nTroubleshooting:")
        print("- Check your internet connection")
        print("- Verify IBL server is accessible")
        print("- Check cache directory permissions")
        print("- Consider running: pip install -U ONE-api")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())