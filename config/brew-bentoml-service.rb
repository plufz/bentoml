class BentomlMultiservice < Formula
  desc "BentoML Multi-Service AI API server"
  homepage "https://github.com/plufz/bentoml"
  url "file:///Users/emiledeholt/Projects/Repositories/bentoml"
  version "1.0.0"
  
  depends_on "python@3.11"
  depends_on "uv"
  
  def install
    # Install the service files to the prefix
    prefix.install Dir["*"]
    
    # Create a wrapper script that sets up the environment and runs the service
    (bin/"bentoml-multiservice").write <<~EOS
      #!/bin/bash
      cd #{prefix}
      export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
      exec #{prefix}/scripts/start.sh
    EOS
  end
  
  service do
    run [opt_bin/"bentoml-multiservice"]
    working_dir HOMEBREW_PREFIX
    log_path var/"log/bentoml-multiservice.log"
    error_log_path var/"log/bentoml-multiservice.error.log"
    environment_variables PATH: std_service_path_env
    keep_alive true
    process_type :adaptive
  end
  
  test do
    # Basic test to ensure the service can start (but not actually run it)
    assert_predicate bin/"bentoml-multiservice", :exist?
    assert_predicate bin/"bentoml-multiservice", :executable?
  end
end