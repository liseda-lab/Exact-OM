.PHONY: build-frontend

build-frontend:
	cd explanations_visualizer && npm ci --no-audit --no-fund && npm run build
	rm -rf exact_inspect/static
	mkdir -p exact_inspect/static
	cp -R explanations_visualizer/out/. exact_inspect/static/
