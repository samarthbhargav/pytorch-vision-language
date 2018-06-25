angular.module('gveApp', [])
    .directive('compile', ['$compile', function ($compile) {
        return function(scope, element, attrs) {
            scope.$watch(
                function(scope) {
                    // watch the 'compile' expression for changes
                    return scope.$eval(attrs.compile);
                },
                function(value) {
                    // when the 'compile' expression changes
                    // assign it into the current DOM
                    element.html(value);

                    // compile the new DOM and link it to the current
                    // scope.
                    // NOTE: we only compile .childNodes so that
                    // we don't get into infinite loop compiling ourselves
                    $compile(element.contents())(scope);
                }
            );
        };
    }])
    .controller('MainCtrl', ['$scope', '$http', function ($scope, $http) {

        $scope.loading_fact_explainer = false;
        $scope.loading_counterfact_explainer = true;
        $scope.server = "http://localhost:5000";
        $scope.overlayTracker = [ -1, -1, -1, -1, -1, -1, -1, -1, -1 ];

        $scope.overlay = function(image_index, overlay_index) {
            $scope.overlayTracker[image_index] = overlay_index;
        };

        $scope.addOverlayAnchors = function (image, overlay_tracker_index) {
            var displayText = image.display;
            image.overlays = [];
            image.links = wordHighlightsToLinks(image.word_highlights);
            var i = 1;
            for (var link in image.links) {
                var parts = displayText.split(link);
                if (parts.length == 2) {
                    displayText = parts[0] + "<span ng-mouseleave=\"overlay(" + overlay_tracker_index + ", -1)\" ng-mouseover=\"overlay(" + overlay_tracker_index + ", " + i + ")\" class='anchor anchor" + i++ + "'>" + link + "</span>" + parts[1];
                    image.overlays.push(image.links[link]);
                }
            }
            image.display = displayText;
            return image;
        };

        $scope.displayExplanation = function (image, counter) {
            counter = counter || false;
            var classLabel = image.class_label;
            if(counter) {
                var index = $scope.images.indexOf(image);
                var labelIndex = (index + 1) % 2;
                classLabel = $scope.images[labelIndex].class_label;
            }
            image.display = "This is" + (counter ? " not " : " ") + "a <span class='class_label'>" + classLabel.split("_").join(" ") + "</span> because " + image.explanation;
            return image;
        };

        $scope.explain = function (image_index) {
            $scope.attack_image = null;
            $scope.current = image_index;
            $scope.sample_explanation = "Loading...";
            $http.get($scope.server + '/explain/' + $scope.sample_images[$scope.current].id).then(function (response) {
                $scope.sample_images[$scope.current] = response.data;
                $scope.sample_explanation = $scope.displayExplanation($scope.sample_images[$scope.current]).display;
		$scope.loadingHighlights = true;
                $http.get($scope.server + '/explain/' + $scope.sample_images[$scope.current].id + "?word_highlights=True").then(function (response) {
                    $scope.sample_images[$scope.current] = response.data;
                    $scope.sample_explanation = $scope.addOverlayAnchors($scope.displayExplanation($scope.sample_images[$scope.current]), 0).display;
		    $scope.loadingHighlights = false;
                });
            });
        };

        $scope.attack = function () {
            $scope.attacking = true;
            $scope.attack_explanation = "Attacking...";
            $http.get($scope.server + '/attack/' + $scope.sample_images[$scope.current].id).then(function (response) {
                $scope.attack_image = response.data;
                $scope.attack_image.class_label = $scope.sample_images[$scope.current].class_label;
                $scope.attack_image.explanation = $scope.attack_image.adv_explanation;
                $scope.attack_image.word_highlights = $scope.attack_image.adv_word_highlights;
                $scope.attack_explanation = $scope.addOverlayAnchors($scope.displayExplanation($scope.attack_image), 1).display;
                $scope.attacking = false;
            });
        };

        $scope.load_sample_images = function () {
            $scope.attack_image = null;
            $scope.sample_images = [];
            $http.get($scope.server + '/sample_images/10').then(function (response) {
                $scope.sample_images = response.data;
                $scope.explain(0);
            });
        };

        $scope.load_classes = function () {
            $scope.classes = [];
            $http.get($scope.server + '/classes').then(function (response) {
                $scope.classes = response.data;
                $scope.retry();
            });
        };

        $scope.retry = function () {
            $scope.loading_counterfact_explainer = true;
            $scope.explanation = "";
            $scope.images = [];
            $scope.correct_class_label = "";
            $scope.result = false;
            var class1 = $scope.classes[Math.floor(Math.random() * $scope.classes.length)];
            var index = $scope.classes.indexOf(class1);
            var classes = angular.copy($scope.classes);
            classes.splice(index, 1);
            var class2 = classes[Math.floor(Math.random() * classes.length)];
            $http.get($scope.server + '/counter_factual' + '/' + class1 + '/' + class2).then(function (response) {
                $scope.explanation = response.data.images[0].explanation;
                $scope.correct_class_label = response.data.images[0].class_label;
                $scope.images = response.data.images;
                $scope.images[1].explanation = response.data.cf_explanation;
                $scope.images[0].word_highlights = response.data.cf_attributes;
                $scope.images[1].word_highlights = response.data.cf_attributes;
                $scope.images = shuffle($scope.images);
                $scope.loading_counterfact_explainer = false;
                $scope.answered = false;
                $scope.result = false;
            });
        };

        $scope.load_sample_images();
        $scope.load_classes();

        function shuffle(array) {
            var currentIndex = array.length, temporaryValue, randomIndex;

            // While there remain elements to shuffle...
            while (0 !== currentIndex) {

                // Pick a remaining element...
                randomIndex = Math.floor(Math.random() * currentIndex);
                currentIndex -= 1;

                // And swap it with the current element.
                temporaryValue = array[currentIndex];
                array[currentIndex] = array[randomIndex];
                array[randomIndex] = temporaryValue;
            }

            return array;
        }

        function wordHighlightsToLinks(word_highlights) {
            var links = {};
            for (var word_highlight_index in word_highlights) {
                var word_highlight = word_highlights[word_highlight_index];
                links[word_highlight.word] = word_highlight.mask;
            }
            return links;
        }

        $scope.select = function (image) {
            $scope.answered = true;
            $scope.result = $scope.correct_class_label == image.class_label;
        };
    }]);
