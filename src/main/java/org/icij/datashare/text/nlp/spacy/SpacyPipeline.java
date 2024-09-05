package org.icij.datashare.text.nlp.spacy;

import static java.lang.Thread.sleep;
import static java.util.UUID.randomUUID;
import static org.icij.datashare.json.JsonObjectMapper.MAPPER;

import com.google.inject.Inject;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.icij.datashare.PropertiesProvider;
import org.icij.datashare.asynctasks.Task;
import org.icij.datashare.asynctasks.TaskManager;
import org.icij.datashare.asynctasks.bus.amqp.TaskError;
import org.icij.datashare.text.Language;
import org.icij.datashare.text.nlp.AbstractPipeline;
import org.icij.datashare.text.nlp.NlpTag;
import org.icij.datashare.user.User;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public final class SpacyPipeline extends AbstractPipeline {
    private static final Logger logger = LoggerFactory.getLogger(SpacyPipeline.class);

    private final PropertiesProvider propertiesProvider;

    private final TaskManager taskManager;

    private final int taskPollIntervalS;

    private static final HashMap<Language, String> models = readModels();

    private static final Set<Task.State> READY_STATES = Set.of(Task.State.DONE, Task.State.ERROR, Task.State.CANCELLED);

    private static final String SPACY_NER_TASK_NAME = "spacy-ner";

    private static final Map<String, String> DEFAULT_SPACY_PROPERTIES_MAP = Map.of(
        "spacyNlpTaskPollIntervalS", "1"
    );

    @Inject
    public SpacyPipeline(final PropertiesProvider propertiesProvider, TaskManager taskManager) {
        super(propertiesProvider.getProperties());
        this.propertiesProvider = propertiesProvider;
        Properties spacyDefaultProps = new Properties();
        spacyDefaultProps.putAll(DEFAULT_SPACY_PROPERTIES_MAP.entrySet().stream()
            .filter(entry -> !entry.getValue().isEmpty())
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
        this.propertiesProvider.mergeWith(spacyDefaultProps);
        this.taskManager = taskManager;
        this.taskPollIntervalS = Integer.parseInt(
            String.valueOf(this.propertiesProvider.get("spacyNlpTaskPollIntervalS"))
        );
    }

    @Override
    public Set<Language> supportedLanguages() {
        return models.keySet();
    }

    @Override
    public List<List<NlpTag>> processText(Stream<String> batch, Language language) {
        // TODO: we could apply some caching here by generating the task ID based on the content
        //  (not sure it's useful)
        String taskId = generateTaskId();
        Map<String, Object> args = Map.of("texts", batch.toList(), "language", language.iso6391Code());
        Task<List<NlpTag>> task = new Task<>(taskId, SPACY_NER_TASK_NAME, User.nullUser(), args);
        try {
            taskManager.startTask(task);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return this.pollTask(taskId, this.taskPollIntervalS);
    }

    @Override
    public void terminate(Language language) throws InterruptedException {
        super.terminate(language);
    }

    @Override
    public Optional<String> getPosTagSet(Language language) {
        return Optional.ofNullable(models.get(language));
    }

    private <R> R pollTask(String taskId, int intervalS) {
        Task<R> task = null;
        while (task == null || !READY_STATES.contains(task.getState())) {
            task = taskManager.getTask(taskId);
            double progress = Optional.of(task.getProgress()).orElse(0.0);
            logger.debug(
                "Task(id=\"{}\", status={}, progress={})",
                task.id, task.getState(), String.format("%.2f", progress)
            );
            try {
                sleep(intervalS);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        logger.info(
            "Task(id=\"{}\") has status {}, stopped polling", task.id, task.getState());
        switch (task.getState()) {
            case DONE: {
                return task.getResult();
            }
            case ERROR: {
                logger.error("Task(id=\"{}\") failed", task.id);
                TaskError error = task.getError();
                String msg = "Task(id=\"" + taskId + "\") failed with the following cause(s):\n"
                    + error.toString();
                throw new RuntimeException(msg);
            }
            case CANCELLED: {
                logger.error("task {} was cancelled !", task.id);
                throw new RuntimeException("Task(id=\"" + taskId + "\") was cancelled");
            }
            default:
                throw new IllegalArgumentException(
                    "Task(id=\"{}\") unexpected state: " + task.getState());
        }
    }

    private static HashMap<Language, String> readModels() {
        try (InputStream modelStream = SpacyPipeline.class.getResourceAsStream("models.json")) {
            return (HashMap<Language, String>) MAPPER.readValue(modelStream, HashMap.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private String generateTaskId() {
        return SPACY_NER_TASK_NAME + "-" + randomUUID();
    }
}
